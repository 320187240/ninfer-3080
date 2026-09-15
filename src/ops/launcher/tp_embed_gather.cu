// ninfer::ops - device kernels for vocabulary-parallel embedding addressing.
#include "ops/launcher/tp_embed_gather.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kShardBlock = 128;
constexpr unsigned int kZeroRows    = 32;
constexpr unsigned int kZeroColumns = 8;

// local[t] = 0 <= ids[t] - base < owned ? ids[t] - base : 0
__global__ void embedding_shard_ids_kernel(const std::int32_t* ids, std::int32_t* local_ids,
                                           std::int64_t base, std::int32_t owned,
                                           std::int32_t tokens) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= tokens) { return; }
    const std::int64_t offset = static_cast<std::int64_t>(ids[t]) - base;
    local_ids[t]              = offset >= 0 && offset < owned ? static_cast<std::int32_t>(offset)
                                                              : 0;
}

// out[:,t] = 0 where the global id is outside this rank's row range.
__global__ void embedding_zero_unowned_kernel(const std::int32_t* ids, __nv_bfloat16* out,
                                              std::int32_t hidden, std::int64_t base,
                                              std::int32_t owned, std::int32_t tokens) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) +
                       static_cast<int>(threadIdx.y);
    if (column >= tokens) { return; }
    const std::int64_t offset = static_cast<std::int64_t>(ids[column]) - base;
    if (offset >= 0 && offset < owned) { return; }
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += static_cast<int>(blockDim.x)) {
        out[static_cast<std::int64_t>(d) + static_cast<std::int64_t>(column) * hidden] =
            __float2bfloat16(0.0F);
    }
}

} // namespace

void embedding_shard_ids_launch(const std::int32_t* ids, std::int32_t* local_ids,
                                std::int64_t vocab_base, std::int32_t owned_rows,
                                std::int32_t tokens, cudaStream_t stream) {
    const unsigned int blocks =
        (static_cast<unsigned int>(tokens) + kShardBlock - 1) / kShardBlock;
    embedding_shard_ids_kernel<<<blocks, kShardBlock, 0, stream>>>(ids, local_ids, vocab_base,
                                                                   owned_rows, tokens);
}

void embedding_zero_unowned_launch(const std::int32_t* ids, __nv_bfloat16* out,
                                   std::int32_t hidden, std::int64_t vocab_base,
                                   std::int32_t owned_rows, std::int32_t tokens,
                                   cudaStream_t stream) {
    const unsigned int columns =
        (static_cast<unsigned int>(tokens) + kZeroColumns - 1) / kZeroColumns;
    dim3 block(kZeroRows, kZeroColumns);
    embedding_zero_unowned_kernel<<<columns, block, 0, stream>>>(ids, out, hidden, vocab_base,
                                                                 owned_rows, tokens);
}

} // namespace ninfer::ops::detail
