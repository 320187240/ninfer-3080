#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/load/tp_shard.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::GroupwiseIntW8Endpoints:
    case WeightsProfile::Nvfp4:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape, TpShardContext& shards,
                       artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    if (shape.size() != 2) { throw std::logic_error("weight shape must be [rows,columns]"); }
    const std::int64_t rows    = static_cast<std::int64_t>(*shape.begin());
    const std::int64_t columns = static_cast<std::int64_t>(*std::next(shape.begin()));
    const artifact::TensorSlice* slice = shards.active() ? shards.shard(name) : nullptr;
    std::int64_t effective_rows        = rows;
    std::int64_t effective_columns     = columns;
    if (slice != nullptr) {
        switch (slice->kind) {
        case artifact::TensorSliceKind::Whole:
            break;
        case artifact::TensorSliceKind::Rows: {
            effective_rows = 0;
            for (const auto& range : slice->rows) { effective_rows += range.count; }
            break;
        }
        case artifact::TensorSliceKind::Columns:
            effective_columns = slice->column_count();
            break;
        }
    }
    if (effective_rows <= 0 || effective_rows > rows || effective_columns <= 0 ||
        effective_columns > columns) {
        throw std::logic_error("tp shard resolves an invalid weight geometry for " +
                               std::string(name));
    }
    return WeightPlan{.object        = artifact::bind_tensor(binder, name, format, shape,
                                                             placement, slice),
                      .format        = format,
                      .shape_rows    = static_cast<std::int32_t>(effective_rows),
                      .shape_columns = static_cast<std::int32_t>(effective_columns)};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
    }

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));

    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, plan.gate_up.shape_rows,
                                      plan.gate_up.shape_columns);
    out.down    = materialized_weight(materialized, plan.down, plan.down.shape_rows,
                                      plan.down.shape_columns);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key,
                                             split->query_key.shape_rows,
                                             split->query_key.shape_columns),
            .gate_value = materialized_weight(materialized, split->gate_value,
                                              split->gate_value.shape_rows,
                                              split->gate_value.shape_columns),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, 14336, 5120),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key,
                                             split->query_key.shape_rows,
                                             split->query_key.shape_columns),
            .value_z   = materialized_weight(materialized, split->value_z, split->value_z.shape_rows,
                                             split->value_z.shape_columns),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, 16384, 5120),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, TpShardContext& shards,
                                BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        const auto bind_dense    = [&](std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape) {
            const artifact::TensorSlice* slice = shards.active() ? shards.shard(name) : nullptr;
            return artifact::bind_device_tensor(binder, name, format, shape, slice);
        };
        target.input_norm = bind_dense(prefix + "input_norm", NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {7168, 5120}, shards),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {7168, 5120}, shards),
            };
            target.attention.query_norm =
                bind_dense(prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm =
                bind_dense(prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {5120, 6144}, shards);
        } else {
            target.gdn.a_log   = bind_dense(prefix + "gdn/a_log", NumericFormat::FP32, {48});
            target.gdn.dt_bias = bind_dense(prefix + "gdn/dt_bias", NumericFormat::FP32, {48});
            {
                const artifact::TensorSlice* conv_slice =
                    shards.active() ? shards.shard(prefix + "gdn/convolution") : nullptr;
                target.gdn.convolution = artifact::bind_device_tensor(
                    binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240},
                    conv_slice);
                target.gdn.convolution_columns =
                    conv_slice != nullptr &&
                            conv_slice->kind == artifact::TensorSliceKind::Columns
                        ? static_cast<std::int32_t>(conv_slice->column_count())
                        : 10240;
            }
            target.gdn.a_projection =
                bind_dense(prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection =
                bind_dense(prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 5120}, shards),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {12288, 5120}, shards),
            };
            target.gdn.norm = bind_dense(prefix + "gdn/norm", NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S,
                            {5120, 6144}, shards);
        }
        target.post_attention_norm =
            bind_dense(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S,
                                         {34816, 5120}, shards);
        target.mlp.down = bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S,
                                      {5120, 17408}, shards);
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, TpShardContext& shards, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        const auto bind_dense    = [&](std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape) {
            const artifact::TensorSlice* slice = shards.active() ? shards.shard(name) : nullptr;
            return artifact::bind_device_tensor(binder, name, format, shape, slice);
        };
        target.input_norm = bind_dense(prefix + "input_norm", NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            WeightPlan input;
            if (is_early_attention_input(layer)) {
                input = bind_weight(binder, prefix + "attention/query_key_gate_value",
                                    NumericFormat::BF16, {14336, 5120}, shards);
            } else {
                input = bind_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value", 14336, 5120,
                    prefix + "attention/input_projection/input_scale_divisor");
            }
            target.attention.projection =
                FusedAttentionProjectionPlan{.query_key_gate_value = input};
            target.attention.query_norm =
                bind_dense(prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm =
                bind_dense(prefix + "attention/key_norm", NumericFormat::BF16, {256});
            if (is_bf16_attention_output(layer)) {
                target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                      NumericFormat::BF16, {5120, 6144}, shards);
            } else {
                target.attention.output =
                    bind_nvfp4_weight(binder, prefix + "attention/output", 5120, 6144,
                                      prefix + "attention/output_projection/input_scale_divisor");
            }
        } else {
            target.gdn.a_log   = bind_dense(prefix + "gdn/a_log", NumericFormat::FP32, {48});
            target.gdn.dt_bias = bind_dense(prefix + "gdn/dt_bias", NumericFormat::FP32, {48});
            {
                const artifact::TensorSlice* conv_slice =
                    shards.active() ? shards.shard(prefix + "gdn/convolution") : nullptr;
                target.gdn.convolution = artifact::bind_device_tensor(
                    binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240},
                    conv_slice);
                target.gdn.convolution_columns =
                    conv_slice != nullptr &&
                            conv_slice->kind == artifact::TensorSliceKind::Columns
                        ? static_cast<std::int32_t>(conv_slice->column_count())
                        : 10240;
            }
            target.gdn.a_projection =
                bind_dense(prefix + "gdn/a_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.b_projection =
                bind_dense(prefix + "gdn/b_projection", NumericFormat::BF16, {48, 5120});
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", 16384, 5120,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = bind_dense(prefix + "gdn/norm", NumericFormat::BF16, {128});
            if (is_bf16_gdn_output(layer)) {
                target.gdn.output = bind_weight(binder, prefix + "gdn/output",
                                                NumericFormat::BF16, {5120, 6144}, shards);
            } else {
                target.gdn.output =
                    bind_nvfp4_weight(binder, prefix + "gdn/output", 5120, 6144,
                                      prefix + "gdn/output_projection/input_scale_divisor");
            }
        }
        target.post_attention_norm =
            bind_dense(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features, std::uint8_t tp_rank,
                               bool tp_active) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    TpShardContext shards(tp_rank, tp_active);
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {248320, 5120}, shards);    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        bind_groupwise_text_layers(binder, shards, out);
        break;
    case WeightsProfile::Nvfp4:
        bind_nvfp4_text_layers(binder, shards, out);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {5120});
    out.output_head =
        bind_weight(binder, "text/output_head", vocabulary_format, {248320, 5120}, shards);
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    const auto* draft_ids_slice =
        shards.active() ? shards.shard("text/draft_head_token_ids") : nullptr;
    out.draft_head = bind_weight(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                 {131072, 5120}, shards, proposal_placement);
    out.draft_head_token_ids =
        artifact::bind_tensor(binder, "text/draft_head_token_ids", NumericFormat::I32, {131072},
                              proposal_placement, draft_ids_slice);
    out.draft_head_token_count =
        draft_ids_slice != nullptr && draft_ids_slice->kind == artifact::TensorSliceKind::Rows
            ? static_cast<std::int32_t>(draft_ids_slice->row_count())
            : 131072;
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp_dense = [&](std::string_view name, NumericFormat format,
                                    std::initializer_list<std::uint64_t> shape) {
        const artifact::TensorSlice* slice = shards.active() ? shards.shard(name) : nullptr;
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement, slice);
    };
    const auto bind_mtp_weight = [&](std::string_view name, NumericFormat format,
                                     std::initializer_list<std::uint64_t> shape) {
        return bind_weight(binder, name, format, shape, shards, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp_weight("mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm       = bind_mtp_dense("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm          = bind_mtp_dense("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm           = bind_mtp_dense("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp_weight(
        "mtp/layer/attention/query_key_gate_value", NumericFormat::W8G32_F16S, {14336, 5120});
    out.mtp.query_norm =
        bind_mtp_dense("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm = bind_mtp_dense("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp_weight("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp_dense("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up =
        bind_mtp_weight("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120});
    out.mtp.mlp.down =
        bind_mtp_weight("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {5120, 17408});
    out.mtp.final_norm = bind_mtp_dense("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding,
                                              plan.token_embedding.shape_rows,
                                              plan.token_embedding.shape_columns);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = materialized_weight(backing, source.attention.output,
                                             source.attention.output.shape_rows,
                                             source.attention.output.shape_columns);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16,
                {source.gdn.convolution_columns, 4});
            target.projection.a_projection = artifact::materialized_weight(
                backing, source.gdn.a_projection, NumericFormat::BF16, 48, 5120);
            target.projection.b_projection = artifact::materialized_weight(
                backing, source.gdn.b_projection, NumericFormat::BF16, 48, 5120);
            target.projection.input_projection = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output = materialized_weight(backing, source.gdn.output, source.gdn.output.shape_rows,
                                         source.gdn.output.shape_columns);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = materialized_weight(backing, plan.output_head, plan.output_head.shape_rows,
                                      plan.output_head.shape_columns);
    if (plan.features.optimized_proposal()) {
        auto& proposal = runtime.optimized_proposal.emplace();
        proposal.head  = materialized_weight(backing, plan.draft_head, plan.draft_head.shape_rows,
                                             plan.draft_head.shape_columns);
        proposal.token_ids = artifact::materialized_tensor(
            backing, plan.draft_head_token_ids, NumericFormat::I32,
            {plan.draft_head_token_count});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = materialized_weight(
            backing, plan.mtp.input_projection, plan.mtp.input_projection.shape_rows,
            plan.mtp.input_projection.shape_columns);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed = materialized_weight(
            backing, plan.mtp.query_key_gate_value,
            plan.mtp.query_key_gate_value.shape_rows, plan.mtp.query_key_gate_value.shape_columns);
        // The packed projection holds [query|key|gate|value] with six query rows per key/value
        // row pair; the shard gathers half of every block, so the views derive from the
        // effective packed rows (14336 full / 7168 per rank).
        const std::int32_t packed_rows = plan.mtp.query_key_gate_value.shape_rows;
        const std::int32_t kv_rows     = packed_rows / 14;
        const std::int32_t q_rows      = 6 * kv_rows;
        mtp.attention.query       = row_view(mtp.attention.packed, 0, q_rows);
        mtp.attention.key         = row_view(mtp.attention.packed, q_rows, kv_rows);
        mtp.attention.output_gate = row_view(mtp.attention.packed, q_rows + kv_rows, q_rows);
        mtp.attention.value       = row_view(mtp.attention.packed, 2 * q_rows + kv_rows, kv_rows);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output =
            materialized_weight(backing, plan.mtp.output, plan.mtp.output.shape_rows,
                                plan.mtp.output.shape_columns);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
