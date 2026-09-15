#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.
//
// Per-rank execution context for two-GPU tensor parallelism. A TpProgram coordinator
// constructs one per-rank Program of a shard-shaped Variant instantiation and hands each
// rank's schedule this context; every hook below is inert when the TextContext carries a
// null pointer, which is the single-GPU default.

#include "core/device.h"
#include "core/tensor.h"
#include "core/tp_link.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/tp_embedding.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Mirrored host-side facts for one rank's schedule. All device buffers are owned by the
// coordinator, allocated once at startup, and live for the engine's lifetime.
struct TpExec {
    int side                     = 0;  // TpLink side, identical to this rank's shard index
    TpLink* link                 = nullptr;
    std::int32_t hidden          = 0;  // full (unsplit) hidden width
    std::int32_t vocab_rows      = 0;  // this rank's embedding/lm_head rows
    std::int64_t vocab_base      = 0;  // side * vocab_rows
    std::int32_t max_tokens      = 0;  // prefill chunk bound for the scratch buffers
    __nv_bfloat16* zero_residual = nullptr; // side 1 only: [max_tokens * hidden]
    std::int32_t* local_ids      = nullptr; // [max_tokens]
};

// x = sum of both ranks' x (bf16 partials, fp32 accumulate) on both ranks.
inline void tp_allreduce_hidden(TpExec& tp, const Tensor& x, std::int32_t tokens,
                                cudaStream_t stream) {
    tp.link->allreduce_add_side(tp.side, stream, static_cast<__nv_bfloat16*>(x.data),
                                static_cast<std::size_t>(tokens) * tp.hidden);
}

// Row-parallel leaf target: side 1 accumulates into a persistent zero scratch; side 0
// accumulates into the residual stream itself. The commit's allreduce adds the residual
// exactly once on side 0 and the scratch's zero baseline on side 1, landing the full sum
// in the residual stream on BOTH ranks directly (the split-destination exchange), so side
// 1 needs no copy-back node. The scratch is zeroed once at startup and returned to zero
// inside every commit exchange, keeping captured replays self-contained.
inline Tensor tp_row_parallel_target(TpExec& tp, const Tensor& residual, std::int32_t tokens,
                                     cudaStream_t stream) {
    if (tp.side == 0) { return residual; }
    if (residual.ne[0] != tp.hidden || residual.ne[1] != tokens) {
        throw std::invalid_argument("TP row-parallel residual does not match the token window");
    }
    if (tokens > tp.max_tokens) {
        throw std::invalid_argument("TP row-parallel window exceeds the startup scratch");
    }
    return Tensor(tp.zero_residual, DType::BF16, {tp.hidden, tokens});
}

inline void tp_row_parallel_commit(TpExec& tp, Tensor& residual, const Tensor& target,
                                   std::int32_t tokens, cudaStream_t stream) {
    const int clear_src = tp.side == 0 ? 0 : 1;
    tp.link->allreduce_add_side(tp.side, stream,
                                static_cast<const __nv_bfloat16*>(target.data),
                                static_cast<__nv_bfloat16*>(residual.data),
                                static_cast<std::size_t>(tokens) * tp.hidden, clear_src);
}

// Vision tower transport: the tower runs on rank 0 only (the two ranks' GPUs would produce
// FP-reduction-order-divergent embeddings, and the mirrored text prefill needs bit-identical
// image-position hiddens), so rank 0 contributes its encoded embeddings in place while rank 1
// contributes the persistent zero scratch — the fp32-accumulate sum is exactly rank 0's bf16
// values (x+0) on both ranks. The item output is sliced to the prefill-sized exchange bound.
inline void tp_vision_broadcast(TpExec& tp, const Tensor& embeddings, cudaStream_t stream) {
    if (embeddings.dtype != DType::BF16 || embeddings.ne[0] != tp.hidden ||
        embeddings.ne[2] != 1 || embeddings.ne[3] != 1 || !embeddings.is_contiguous() ||
        embeddings.data == nullptr) {
        throw std::invalid_argument("TP vision broadcast embeddings are invalid");
    }
    const std::int64_t tokens = embeddings.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("TP vision broadcast has no tokens"); }
    auto* out = static_cast<__nv_bfloat16*>(embeddings.data);
    const std::size_t elements   = static_cast<std::size_t>(tokens) * tp.hidden;
    const std::size_t slice      = static_cast<std::size_t>(tp.max_tokens) * tp.hidden;
    const __nv_bfloat16* zeros   = tp.zero_residual;
    if (tp.side == 1 && zeros == nullptr) {
        throw std::logic_error("TP rank 1 vision broadcast has no zero scratch");
    }
    for (std::size_t offset = 0; offset < elements; offset += slice) {
        const std::size_t n = std::min(slice, elements - offset);
        if (tp.side == 0) {
            tp.link->allreduce_add_side(0, stream, out + offset, out + offset, n, 0);
        } else {
            tp.link->allreduce_add_side(1, stream, zeros, out + offset, n, 0);
        }
    }
}

// Vocab-parallel embedding lookup: this rank contributes only its owned rows and zeroes the
// rest, then the allreduce yields the full hidden on both ranks.
inline void tp_embedding_gather(TpExec& tp, const Tensor& ids, const Weight& table, Tensor& out,
                                std::int32_t tokens, cudaStream_t stream) {
    if (ids.ne[0] != tokens || out.ne[0] != tp.hidden || out.ne[1] != tokens) {
        throw std::invalid_argument("TP embedding shapes do not match the token window");
    }
    if (tokens > tp.max_tokens) {
        throw std::invalid_argument("TP embedding window exceeds the startup scratch");
    }
    Tensor local_ids(tp.local_ids, DType::I32, {tokens});
    ops::embedding_shard_ids(ids, local_ids, tp.vocab_base, tp.vocab_rows, stream);
    ops::embedding(local_ids, table, out, stream);
    ops::embedding_zero_unowned(ids, out, tp.vocab_base, tp.vocab_rows, stream);
    tp_allreduce_hidden(tp, out, tokens, stream);
}

// Vocab-parallel greedy argmax over this rank's lm_head slice; the reduced global winner
// lands in out_token on both ranks with identical tie-breaking.
inline void tp_argmax_sample(TpExec& tp, const Tensor& logits_column, Tensor& out_token,
                             cudaStream_t stream) {
    if (logits_column.dtype != DType::BF16 || logits_column.ne[0] != tp.vocab_rows ||
        logits_column.ne[1] != 1 || out_token.dtype != DType::I32 || out_token.ne[0] != 1 ||
        out_token.data == nullptr) {
        throw std::invalid_argument("TP argmax sample shapes are invalid");
    }
    tp.link->allreduce_argmax_side(tp.side, stream,
                                   static_cast<const __nv_bfloat16*>(logits_column.data),
                                   static_cast<std::size_t>(tp.vocab_rows),
                                   static_cast<std::int32_t>(tp.vocab_base),
                                   static_cast<std::int32_t*>(out_token.data));
}

// Multi-candidate vocab-parallel greedy argmax: logits is [slice_rows, rows] (one slice per
// candidate column; slice_rows is this rank's lm_head rows, or its shortlist-head rows when
// id_map is set) and every row's winner lands in out_tokens on both ranks in one exchange.
// id_map, when non-null, is this rank's shortlist remap half: the local winner position is
// mapped to the global token id before the comparison, so MTP proposals reduce
// (value, global id) exactly like the full lm_head path.
inline void tp_argmax_sample_rows(TpExec& tp, const Tensor& logits, std::int32_t rows,
                                  Tensor& out_tokens, cudaStream_t stream,
                                  const std::int32_t* id_map = nullptr) {
    const std::int64_t slice_rows = logits.ne[0];
    if (logits.dtype != DType::BF16 || slice_rows <= 0 || slice_rows > tp.vocab_rows || rows <= 0 ||
        logits.numel() != slice_rows * rows || out_tokens.dtype != DType::I32 ||
        out_tokens.numel() != rows || out_tokens.data == nullptr) {
        throw std::invalid_argument("TP argmax sample rows shapes are invalid");
    }
    tp.link->allreduce_argmax_rows_side(
        tp.side, stream, static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<std::size_t>(slice_rows), static_cast<std::size_t>(slice_rows), rows,
        static_cast<std::int32_t>(tp.vocab_base), id_map,
        static_cast<std::int32_t*>(out_tokens.data));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
