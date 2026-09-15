#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_6::detail {
namespace qwen3_6_27b_tp_runtime {
// Forward declaration of the TP coordinator defined by impl/tp_variant.cpp's family
// instantiation; only the Package aliases and the registry need the name.
template <class>
class TpProgram;
} // namespace qwen3_6_27b_tp_runtime
} // namespace targets::qwen3_6::detail

namespace targets::qwen3_6_27b {

struct Package;

namespace detail {

struct Variant;
struct TpVariant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
    GroupwiseIntW8Endpoints,
    Nvfp4,
};

using Frontend       = qwen3_6::Frontend;
using PreparedPrompt = qwen3_6::PreparedPrompt;
using OutputSession  = qwen3_6::OutputSession;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_27b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id           = "qwen3.6-27b";
    static constexpr std::string_view target_key         = "qwen3_6_27b";
    static constexpr std::string_view qwen3_8_model_id   = "qwen3.8-27b";
    static constexpr std::string_view qwen3_8_target_key = "qwen3_8_27b";

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = qwen3_6::SequencePlanner<detail::Variant>;
    using SequencePlan    = qwen3_6::SequencePlan<detail::Variant>;
    using RequestBasePlan = qwen3_6::RequestBasePlan<detail::Variant>;
    using RequestPlan     = qwen3_6::RequestPlan<detail::Variant>;
    using Program         = qwen3_6::Program<detail::Variant>;

    // Two-rank tensor-parallel surface: one shard-shaped Variant instantiation with its own
    // family namespace, coordinated by TpProgram (impl/tp_variant.cpp).
    using TpVariant         = detail::TpVariant;
    using TpSequencePlanner = qwen3_6::SequencePlanner<detail::TpVariant>;
    using TpSequencePlan    = qwen3_6::SequencePlan<detail::TpVariant>;
    using TpRequestBasePlan = qwen3_6::RequestBasePlan<detail::TpVariant>;
    using TpRequestPlan     = qwen3_6::RequestPlan<detail::TpVariant>;
    using TpProgram         = qwen3_6::detail::qwen3_6_27b_tp_runtime::TpProgram<detail::TpVariant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile, std::uint8_t tp_rank);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
    [[nodiscard]] static TpSequencePlanner make_tp_sequence_planner(DeviceContext& device,
                                                                    const EngineOptions& options,
                                                                    WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<TpProgram>
    create_tp_program(const LoadedModel& model_a, const LoadedModel& model_b,
                      TpSequencePlan&& plan, DeviceContext& device_a, DeviceContext& device_b,
                      int primary_device,
                      TpClockHolderMode clock_holder = TpClockHolderMode::Demand,
                      std::uint32_t clock_holder_hold_ms = 10'000);
};

} // namespace targets::qwen3_6_27b
} // namespace ninfer
