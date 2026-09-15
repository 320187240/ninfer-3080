#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::artifact {
struct TensorSlice;
}

namespace ninfer::targets::qwen3_6_27b::detail {

// M3a two-rank shard contract for the qwen3.6-27b family groupwise-int profiles. Each rank
// materializes exactly the slice named by tp_weight_shard; the tensors it does not own are
// validate-only in its binder so the full inventory stays accounted for. Every slice is an exact
// byte transform of the artifact payload: row-range concatenations are layout-native in
// row-split-k128-v1, and column halves densify one group-aligned K prefix per row.
//
// Destination row order of a repacked tensor follows ascending source-row order, so the M3b
// schedule can recover each block's source rows from the split ranges themselves.
[[nodiscard]] artifact::TensorSlice tp_weight_shard(std::string_view name, std::uint8_t rank);

// The single-GPU placement this shard table is derived from: the whole tensor on every rank.
[[nodiscard]] bool tp_is_duplicated(std::string_view name);

} // namespace ninfer::targets::qwen3_6_27b::detail
