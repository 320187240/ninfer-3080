#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Vocabulary-parallel embedding addressing for two-rank tensor parallelism. The embedding
 * table is row-split: rank r owns global rows [vocab_base, vocab_base + owned_rows).
 *
 * embedding_shard_ids maps each global id to its rank-local table row, clamping unowned ids
 * to row 0 so the plain embedding gather stays in bounds:
 *
 *   local[t] = in [0, owned_rows) ? ids[t] - vocab_base : 0.
 *
 * `ids` is contiguous I32 [T], `local_ids` is contiguous I32 [T] and must not overlap `ids`.
 *
 * embedding_zero_unowned then removes the placeholder rows the clamp introduced:
 *
 *   out[:,t] = 0 where ids[t] is outside [vocab_base, vocab_base + owned_rows).
 *
 * `out` is contiguous BF16 [D,T]; only whole columns are cleared. Composing both with
 * ops::embedding yields each rank's zero-partial hidden, which a TpLink allreduce sums into
 * the full hidden on both ranks.
 */
void embedding_shard_ids(const Tensor& ids, Tensor& local_ids, std::int64_t vocab_base,
                         std::int32_t owned_rows, cudaStream_t stream);

void embedding_zero_unowned(const Tensor& ids, Tensor& out, std::int64_t vocab_base,
                            std::int32_t owned_rows, cudaStream_t stream);

} // namespace ninfer::ops
