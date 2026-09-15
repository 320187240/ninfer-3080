#include "targets/qwen3_6_27b/impl/tp_variant.h"

#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::TpVariant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_tp_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"
#include "targets/qwen3_6/impl/runtime/tp_program.h"

namespace ninfer::targets::qwen3_6_27b::detail {

using ShardConfig = TpTextConfig;

// The leaves below re-implement the single-GPU Variant bodies whose allocation shapes and
// channel views derive from compile-time TextConfig constants; here those constants are the
// rank's shard geometry. Leaves that are fully weight-driven (attention projections,
// o_proj/out_proj linear_add, GDN norm/control) are inherited unchanged.

void TpVariant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({ShardConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat,
                        ops::LinearPolicy::A16Only, workspace, stream);
}

void TpVariant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view =
        output_gate.view({ShardConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, ops::LinearPolicy::A16Only, workspace,
                                      stream);
}

void TpVariant::gdn_input_projection_record(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    // Same split-projection record as the single-GPU leaf, against the rank's shard
    // channels: the conv state and record slots are per-rank pools sized by TpTextConfig.
    auto workspace_scope = workspace.scope();
    const std::size_t record_bytes = std::max<std::size_t>(
        1, ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
               ShardConfig::key_dim, ShardConfig::key_dim, ShardConfig::value_dim, hidden.ne[2],
               hidden.ne[1], hidden.ne[1]));
    const DeviceSpan storage = workspace.alloc_bytes(record_bytes);
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view =
        output_gate.view({ShardConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value,
                                    output_gate_view, ops::LinearPolicy::A16Only, leaf_workspace,
                                    stream);
}

std::size_t TpVariant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size,
    std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
    if (weights_profile == WeightsProfile::Nvfp4) {
        throw std::logic_error("TP shard record leaf is groupwise-int only");
    }
    return std::max<std::size_t>(
        1, ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
               ShardConfig::key_dim, ShardConfig::key_dim, ShardConfig::value_dim, batch_size,
               first, last));
}

void TpVariant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {ShardConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, ops::LinearPolicy::A16Only, workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, ops::LinearPolicy::A16Only, workspace,
                    stream);
}

std::size_t TpVariant::gdn_norm_control_projection_workspace_capacity_bytes(
    std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(2 * ShardConfig::gdn_value_heads,
                                                              ShardConfig::hidden, first, last);
}

// The MTP draft layer's packed attention projection on the rank's gathered shard rows:
// [q_rank; k_rank; gate_rank; v_rank] halves, split by the same head counts the schedule
// uses, so the shared splitter op works unchanged against TpTextConfig geometry.
void TpVariant::mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights, Tensor& query,
                                         Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {ShardConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({ShardConfig::head_dim, ShardConfig::query_heads, cols});
    Tensor key_heads   = key.view({ShardConfig::head_dim, ShardConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({ShardConfig::head_dim, ShardConfig::query_heads, cols});
    Tensor value_heads = value.view({ShardConfig::head_dim, ShardConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

// The fused linear_pair family is registered for the full [1024,K] kv rows; the rank's
// [512,5120] halves run as two plain projections over the same x.
void TpVariant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                  Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void TpVariant::mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

// Row-parallel MTP MLP: gate_up holds the rank's gathered halves and down its column half;
// the schedule hands the leaf the row-parallel target as `residual`, so the accumulated
// partial commits through the TpLink exchange exactly like the target stack's post_mixer.
void TpVariant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {ShardConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {ShardConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, ShardConfig::intermediate),
                  gate_up.slice(0, ShardConfig::intermediate, ShardConfig::intermediate),
                  activation, stream);
    Tensor delta = workspace.alloc(DType::BF16, {ShardConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6_27b {

Package::TpSequencePlanner Package::make_tp_sequence_planner(DeviceContext& device,
                                                             const EngineOptions& options,
                                                             WeightsProfile weights_profile) {
    auto planner =
        qwen3_6::make_sequence_planner<detail::TpVariant>(device, options, weights_profile);
    // Both ranks of the TpProgram coordinator construct their ProgramImpl before the
    // two-rank seams exist, so graph preparation must wait for the coordinator.
    planner.impl_->inputs.defer_graph_capture  = true;
    planner.impl_->minimum->defer_graph_capture = true;
    return planner;
}

std::unique_ptr<Package::TpProgram>
Package::create_tp_program(const LoadedModel& model_a, const LoadedModel& model_b,
                           TpSequencePlan&& plan, DeviceContext& device_a,
                           DeviceContext& device_b, int primary_device,
                           TpClockHolderMode clock_holder, std::uint32_t clock_holder_hold_ms) {
    if (model_a.impl_ == nullptr || model_b.impl_ == nullptr) {
        throw std::invalid_argument("TP loaded model is empty");
    }
    return std::make_unique<Package::TpProgram>(model_a.impl_->data.runtime,
                                                model_b.impl_->data.runtime, std::move(plan),
                                                device_a, device_b, primary_device, clock_holder,
                                                clock_holder_hold_ms);
}

} // namespace ninfer::targets::qwen3_6_27b
