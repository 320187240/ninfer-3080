#pragma once

// ninfer::ops::detail - private launch prototypes for vocabulary-parallel embedding
// addressing (two-rank tensor parallelism).

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void embedding_shard_ids_launch(const std::int32_t* ids, std::int32_t* local_ids,
                                std::int64_t vocab_base, std::int32_t owned_rows,
                                std::int32_t tokens, cudaStream_t stream);

void embedding_zero_unowned_launch(const std::int32_t* ids, __nv_bfloat16* out,
                                   std::int32_t hidden, std::int64_t vocab_base,
                                   std::int32_t owned_rows, std::int32_t tokens,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
