#pragma once

#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6_27b::detail {

// Compile-time dimensions of one rank's shard of the two-GPU tensor-parallel split: half the
// attention heads (12 Q / 2 KV), half the GDN heads (8 K / 24 V) and conv channels, half the
// MLP intermediate, and half the embedding/lm_head vocabulary rows. hidden, layers, head_dim,
// rotary_dim, and the tokenizer token domain stay whole because the schedule runs the same
// token stream on both ranks. Mirrors the split table in impl/load/tp_shard.cpp exactly.
struct TpTextConfig {
    static constexpr int hidden       = 5120;
    static constexpr int layers       = 64;
    static constexpr int intermediate = 8704;

    static constexpr int output_rows  = 124160;
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 8;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 24;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 12;
    static constexpr int kv_heads    = 2;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    static constexpr int full_attention_interval = qwen3_6::kHybridAttentionInterval;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e7F;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = 2 * query_size;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return qwen3_6::is_full_attention_layer(layer);
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        return qwen3_6::full_attention_layers(layers);
    }

    [[nodiscard]] static constexpr int gdn_layers() { return qwen3_6::gdn_layers(layers); }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return qwen3_6::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) { return qwen3_6::gdn_index(layer); }
};

static_assert(TpTextConfig::full_attention_layers() == 16);
static_assert(TpTextConfig::gdn_layers() == 48);
static_assert(TpTextConfig::query_size == TextConfig::query_size / 2);
static_assert(TpTextConfig::query_size + TpTextConfig::kv_size == 3584); // query_key shard rows
static_assert(TpTextConfig::convolution_dim == TextConfig::convolution_dim / 2);
static_assert(TpTextConfig::value_dim == TextConfig::value_dim / 2);
static_assert(TpTextConfig::intermediate == TextConfig::intermediate / 2);
static_assert(TpTextConfig::output_rows == TextConfig::output_rows / 2);

// The exact Variant of the per-rank shard schedules. It reuses the single-GPU Variant's
// weights profile, model view, payload types, and every leaf whose shapes are fully
// weight-driven; the leaves whose bodies derive from compile-time TextConfig constants are
// re-implemented against TpTextConfig in impl/tp_variant.cpp. The MTP draft layer runs with
// the same shard geometry as the target stack (M3a's split table already slices its
// tensors), so its config-dependent leaves are re-implemented here too; the fully
// weight-driven MTP leaves (kv / q-gate projections) and all replay paths are inherited.
struct TpVariant : Variant {
    using TextConfig = detail::TpTextConfig;

    // The rank's shortlist-head rows (half of 131072); drives the proposal scratch plan.
    static constexpr std::int32_t draft_head_rows = 65536;

    // The duplicated GDN a/b control projections run full-width (their kernels are fixed to
    // the registered head count), so gdn_mix stages their whole-model outputs plus the
    // extracted rank halves across the GDN core and the workspace plan must carry them.
    static constexpr std::int32_t gdn_control_full_width_heads = 2 * TextConfig::gdn_value_heads;

    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                           cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights, Tensor& query,
                                         Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                  Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                  cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);
    // The duplicated GDN a/b control projections run full-width (their kernels are fixed to
    // the registered head count), so the leaf workspace is planned for the full heads.
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    // The replay-record GDN input projection runs on the rank's shard channels; the
    // inherited planner would budget the full-width leaf (safe but oversized).
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
};

} // namespace ninfer::targets::qwen3_6_27b::detail
