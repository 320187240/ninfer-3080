#include "ninfer/ops/mtp_pack.h"
#include "ops/launcher/mtp_pack.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, const char* op,
                   const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_pack_fc_input";
    require_bf16_contiguous_nonnull(embedding_norm, op, "embedding_norm");
    require_bf16_contiguous_nonnull(hidden_norm, op, "hidden_norm");
    require_bf16_contiguous_nonnull(out, op, "out");
    const std::int32_t rows   = embedding_norm.ne[0];
    const std::int32_t tokens = embedding_norm.ne[1];
    if (rows <= 0) { throw std::invalid_argument("mtp_pack_fc_input: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("mtp_pack_fc_input: T must be positive"); }
    require_shape(embedding_norm, rows, tokens, op, "embedding_norm");
    require_shape(hidden_norm, rows, tokens, op, "hidden_norm");
    require_shape(out, 2 * rows, tokens, op, "out");

    detail::mtp_pack_fc_input_launch(embedding_norm, hidden_norm, out, stream);
}

void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream) {
    constexpr const char* op = "mtp_split_attn_in";
    require_bf16_contiguous_nonnull(attn_in, op, "attn_in");
    require_bf16_contiguous_nonnull(q, op, "q");
    require_bf16_contiguous_nonnull(k, op, "k");
    require_bf16_contiguous_nonnull(gate, op, "gate");
    require_bf16_contiguous_nonnull(v, op, "v");
    const std::int32_t tokens = attn_in.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_split_attn_in: T must be positive"); }
    if (attn_in.ne[2] != 1 || attn_in.ne[3] != 1) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for attn_in");
    }
    // Registered geometries: the full 27B projection [q 6144 | k 1024 | gate 6144 | v 1024]
    // and the two-rank shard halves [3072 | 512 | 3072 | 512]. Boundaries derive from the
    // concrete q/kv row counts so both domains share one exact index remap.
    const std::int32_t q_rows   = q.ne[0] * q.ne[1];
    const std::int32_t kv_rows  = k.ne[0] * k.ne[1];
    if (q.ne[2] != tokens || q.ne[3] != 1 || gate.ne[2] != tokens || gate.ne[3] != 1 ||
        k.ne[2] != tokens || k.ne[3] != 1 || v.ne[2] != tokens || v.ne[3] != 1 ||
        gate.ne[0] != q.ne[0] || gate.ne[1] != q.ne[1] || v.ne[0] != k.ne[0] ||
        v.ne[1] != k.ne[1] || attn_in.ne[0] != 2 * (q_rows + kv_rows) ||
        (attn_in.ne[0] != 14336 && attn_in.ne[0] != 7168) || q.ne[0] != 256 || k.ne[0] != 256) {
        throw std::invalid_argument("mtp_split_attn_in: invalid shape for the mtp attention split");
    }

    detail::mtp_split_attn_in_launch(attn_in, q, k, gate, v, stream);
}

} // namespace ninfer::ops
