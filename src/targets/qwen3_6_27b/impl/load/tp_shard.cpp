#include "targets/qwen3_6_27b/impl/load/tp_shard.h"

#include "artifact/reader.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::TensorSlice;
using artifact::TensorSliceKind;
using artifact::TensorSliceRange;

// Channel geometry of the split plan: per-rank output channels of each fused projection block
// and the source-row offset where each block starts.
constexpr std::uint64_t kQChannelsHalf      = 3072;  // full-attn Q: 12 of 24 heads
constexpr std::uint64_t kKvChannelsHalf     = 512;   // full-attn K/V: 2 of 4 heads
constexpr std::uint64_t kQkRow             = 6144;  // full-attn query_key/gate_value block split
constexpr std::uint64_t kGdnQChannelsHalf  = 1024;  // GDN Q: 8 of 16 heads
constexpr std::uint64_t kGdnQkRow          = 2048;  // GDN query_key block split
constexpr std::uint64_t kVzChannelsHalf    = 3072;  // GDN V/Z: 24 of 48 heads
constexpr std::uint64_t kVzRow             = 6144;  // GDN value_z block split
constexpr std::uint64_t kMlpChannelsHalf   = 8704;  // MLP gate or up: half of 17408
constexpr std::uint64_t kMlpRow            = 17408; // MLP gate/up block split
constexpr std::uint64_t kVocabHalf         = 124160;
constexpr std::uint64_t kDraftVocabHalf    = 65536;
constexpr std::uint64_t kMtpPackedQGateHalf = 3072; // MTP packed attention q and gate blocks
constexpr std::uint64_t kMtpPackedKvHalf    = 512;  // MTP packed attention k and v blocks

constexpr TensorSlice kWhole{};

TensorSlice rows(std::initializer_list<TensorSliceRange> ranges) {
    TensorSlice slice;
    slice.kind = TensorSliceKind::Rows;
    std::size_t out = 0;
    for (const TensorSliceRange& range : ranges) {
        if (out >= slice.rows.size()) { throw std::logic_error("tp shard uses too many row ranges"); }
        slice.rows[out++] = range;
    }
    return slice;
}

TensorSlice columns(std::uint64_t begin, std::uint64_t count) {
    TensorSlice slice;
    slice.kind            = TensorSliceKind::Columns;
    slice.column_ranges   = {TensorSliceRange{begin, count}};
    return slice;
}

// Disjoint column ranges gathered in order into one densified destination: used where the rank's
// channel order is block-gathered rather than one contiguous source span.
TensorSlice column_gather(std::initializer_list<TensorSliceRange> ranges) {
    TensorSlice slice;
    slice.kind = TensorSliceKind::Columns;
    std::size_t out = 0;
    for (const TensorSliceRange& range : ranges) {
        if (out >= slice.column_ranges.size()) {
            throw std::logic_error("tp shard uses too many column ranges");
        }
        slice.column_ranges[out++] = range;
    }
    return slice;
}

// Two block halves of different channel counts: A rows [rank*half_a, rank*half_a+half_a) at a,
// B rows [rank*half_b, ...) at b. Ranges stay in ascending source-row order for a < b.
TensorSlice block_pair(std::uint8_t rank, std::uint64_t half_a, std::uint64_t a,
                       std::uint64_t half_b, std::uint64_t b) {
    const std::uint64_t offset_a = static_cast<std::uint64_t>(rank) * half_a;
    const std::uint64_t offset_b = static_cast<std::uint64_t>(rank) * half_b;
    return rows({{a + offset_a, half_a}, {b + offset_b, half_b}});
}

// Two block halves with the same channel count.
TensorSlice block_pair(std::uint8_t rank, std::uint64_t half, std::uint64_t a, std::uint64_t b) {
    return block_pair(rank, half, a, half, b);
}

// Suffix of the object name after the layer (or module) prefix it lives under.
std::string_view suffix_of(std::string_view name, std::string_view prefix) {
    if (name.substr(0, prefix.size()) != prefix) { return {}; }
    return name.substr(prefix.size());
}

// Split decision for one Text or MTP layer member, given the member name after the layer prefix.
TensorSlice layer_shard(std::string_view member, std::uint8_t rank, bool full_attention) {
    if (member == "input_norm" || member == "post_attention_norm") { return kWhole; }
    if (full_attention) {
        if (member == "attention/query_key" || member == "attention/gate_value") {
            // Q/KV blocks: query rows [0,6144) hold 12 Q heads per rank (3072 rows), key/value
            // rows [6144,7168) hold 2 KV heads per rank (512 rows).
            return block_pair(rank, kQChannelsHalf, 0, kKvChannelsHalf, kQkRow);
        }
        if (member == "attention/query_norm" || member == "attention/key_norm") { return kWhole; }
        if (member == "attention/output") { return columns(rank * 3072, 3072); }
    } else {
        if (member == "gdn/a_log" || member == "gdn/dt_bias" || member == "gdn/norm" ||
            member == "gdn/a_projection" || member == "gdn/b_projection") {
            return kWhole;
        }
        if (member == "gdn/convolution") {
            // The rank's projected channels are [Q_rank; K_rank; V_rank] (the query_key/value_z
            // row shards gather them in that order), so the per-channel convolution weight must
            // gather the same three channel blocks instead of one contiguous span.
            return column_gather({{rank * kGdnQChannelsHalf, kGdnQChannelsHalf},
                                  {kGdnQkRow + rank * kGdnQChannelsHalf, kGdnQChannelsHalf},
                                  {2 * kGdnQkRow + rank * kVzChannelsHalf, kVzChannelsHalf}});
        }
        if (member == "gdn/query_key") {
            return block_pair(rank, kGdnQChannelsHalf, 0, kGdnQkRow);
        }
        if (member == "gdn/value_z") { return block_pair(rank, kVzChannelsHalf, 0, kVzRow); }
        if (member == "gdn/output") { return columns(rank * 3072, 3072); }
    }
    if (member == "mlp/gate_up") {
        return block_pair(rank, kMlpChannelsHalf, 0, kMlpRow);
    }
    if (member == "mlp/down") { return columns(rank * kMlpChannelsHalf, kMlpChannelsHalf); }
    throw std::logic_error("tp shard table has no entry for text layer member: " +
                           std::string(member));
}

bool is_full_attention_layer(std::size_t layer) {
    return layer >= 3 && (layer - 3) % 4 == 0;
}

} // namespace

artifact::TensorSlice tp_weight_shard(std::string_view name, std::uint8_t rank) {
    if (rank > 1) { throw std::logic_error("tp shard rank is outside {0,1}"); }

    if (name == "text/token_embedding" || name == "text/output_head") {
        return rows({{static_cast<std::uint64_t>(rank) * kVocabHalf, kVocabHalf}});
    }
    if (name == "text/final_norm") { return kWhole; }
    if (name == "text/draft_head" || name == "text/draft_head_token_ids") {
        return rows({{static_cast<std::uint64_t>(rank) * kDraftVocabHalf, kDraftVocabHalf}});
    }
    // The Vision tower and projector duplicate whole on both ranks (tp_shard.h): the tower
    // runs on rank 0 only and its embeddings reach rank 1 through the TpLink broadcast
    // (tp_exec.h), so rank 1's copy is a numerical spare that keeps the binding contract
    // uniform — exactly like text/final_norm, Whole here means the binder places the full
    // tensor with no slice.
    if (name.substr(0, 7) == "vision/") { return kWhole; }

    if (const auto member = suffix_of(name, "mtp/"); !member.empty()) {
        if (member == "input_projection") { return columns(rank * 5120, 5120); }
        if (member == "embedding_norm" || member == "hidden_norm" ||
            member == "layer/input_norm" || member == "layer/post_attention_norm" ||
            member == "final_norm") {
            return kWhole;
        }
        if (member == "layer/attention/query_key_gate_value") {
            // Packed blocks: query [0,6144), key [6144,7168), gate [7168,13312), value
            // [13312,14336). Each rank takes its half of every block, kept in source-row order.
            return rows({
                {static_cast<std::uint64_t>(rank) * kMtpPackedQGateHalf, kMtpPackedQGateHalf},
                {6144 + rank * kMtpPackedKvHalf, kMtpPackedKvHalf},
                {7168 + rank * kMtpPackedQGateHalf, kMtpPackedQGateHalf},
                {13312 + rank * kMtpPackedKvHalf, kMtpPackedKvHalf},
            });
        }
        if (member == "layer/attention/query_norm" || member == "layer/attention/key_norm") {
            return kWhole;
        }
        if (member == "layer/attention/output") { return columns(rank * 3072, 3072); }
        if (member == "layer/mlp/gate_up") {
            return block_pair(rank, kMlpChannelsHalf, 0, kMlpRow);
        }
        if (member == "layer/mlp/down") {
            return columns(rank * kMlpChannelsHalf, kMlpChannelsHalf);
        }
        throw std::logic_error("tp shard table has no entry for mtp member: " + std::string(member));
    }

    constexpr std::string_view kLayerPrefix = "text/layers/";
    if (name.substr(0, kLayerPrefix.size()) == kLayerPrefix) {
        const std::string_view rest = name.substr(kLayerPrefix.size());
        const std::size_t slash     = rest.find('/');
        if (slash == std::string_view::npos || slash == 0 || slash > 2) {
            throw std::logic_error("tp shard table cannot parse text layer name: " +
                                   std::string(name));
        }
        std::size_t layer = 0;
        for (std::size_t i = 0; i < slash; ++i) {
            if (rest[i] < '0' || rest[i] > '9') {
                throw std::logic_error("tp shard table cannot parse text layer index: " +
                                       std::string(name));
            }
            layer = layer * 10 + static_cast<std::size_t>(rest[i] - '0');
        }
        return layer_shard(rest.substr(slash + 1), rank, is_full_attention_layer(layer));
    }

    throw std::logic_error("tp shard table has no entry for artifact object: " + std::string(name));
}

bool tp_is_duplicated(std::string_view name) {
    try {
        const TensorSlice slice = tp_weight_shard(name, 0);
        return slice.kind == TensorSliceKind::Whole;
    } catch (const std::logic_error&) {
        return false;
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
