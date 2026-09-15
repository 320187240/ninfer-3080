// ninfer::ops - vocabulary-parallel embedding addressing for two-rank tensor parallelism.
#include "ninfer/ops/tp_embedding.h"

#include "ops/launcher/tp_embed_gather.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_ids(const Tensor& ids, const char* label) {
    if (ids.dtype != DType::I32 || ids.ne[0] <= 0 || ids.ne[1] != 1 || ids.ne[2] != 1 ||
        ids.ne[3] != 1 || !ids.is_contiguous() || ids.data == nullptr) {
        throw std::invalid_argument(std::string(label) + " requires contiguous I32 [T]");
    }
}

void require_owned_range(std::int64_t vocab_base, std::int32_t owned_rows) {
    if (owned_rows <= 0 || vocab_base < 0) {
        throw std::invalid_argument("embedding shard owned range must be nonnegative");
    }
}

} // namespace

void embedding_shard_ids(const Tensor& ids, Tensor& local_ids, std::int64_t vocab_base,
                         std::int32_t owned_rows, cudaStream_t stream) {
    require_ids(ids, "embedding_shard_ids");
    require_ids(local_ids, "embedding_shard_ids local_ids");
    if (local_ids.ne[0] != ids.ne[0] || local_ids.data == ids.data) {
        throw std::invalid_argument("embedding_shard_ids local_ids must be a distinct [T] tensor");
    }
    require_owned_range(vocab_base, owned_rows);
    detail::embedding_shard_ids_launch(static_cast<const std::int32_t*>(ids.data),
                                       static_cast<std::int32_t*>(local_ids.data), vocab_base,
                                       owned_rows, ids.ne[0], stream);
}

void embedding_zero_unowned(const Tensor& ids, Tensor& out, std::int64_t vocab_base,
                            std::int32_t owned_rows, cudaStream_t stream) {
    require_ids(ids, "embedding_zero_unowned");
    if (out.dtype != DType::BF16 || out.ne[0] <= 0 || out.ne[1] != ids.ne[0] || out.ne[2] != 1 ||
        out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("embedding_zero_unowned requires contiguous BF16 [D,T]");
    }
    require_owned_range(vocab_base, owned_rows);
    detail::embedding_zero_unowned_launch(static_cast<const std::int32_t*>(ids.data),
                                          static_cast<__nv_bfloat16*>(out.data), out.ne[0],
                                          vocab_base, owned_rows, ids.ne[0], stream);
}

} // namespace ninfer::ops
