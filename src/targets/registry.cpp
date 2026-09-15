#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.tp) {
        if (options.tp_devices.size() != 2) {
            throw std::invalid_argument("Engine tp mode requires exactly two tp_devices");
        }
        if (options.tp_devices[0] == options.tp_devices[1]) {
            throw std::invalid_argument("Engine tp_devices must be two distinct devices");
        }
        if (std::find(options.tp_devices.begin(), options.tp_devices.end(), options.device) ==
            options.tp_devices.end()) {
            throw std::invalid_argument("Engine device must be one of tp_devices in tp mode");
        }
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes,
                                                std::string_view device_label) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument(std::string(device_label) + " model weights require " +
                                    std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes = runtime_bytes_after_planned_weights(
        load_plan.materialization().device_capacity_bytes, "primary device");
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model));
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

// M3b two-rank tensor-parallel startup for the qwen3.6/3.8-27B package: each rank binds and
// materializes only its shard, both ranks validate their startup memory, and one shard-shaped
// TpVariant sequence plan resolves KV capacity against the tightest rank before the TpProgram
// coordinator takes over decode/prefill scheduling across both devices.
ConstructedTarget construct_tp_target_27b(const EngineOptions& options, DeviceContext& primary,
                                          const std::vector<int>& tp_devices,
                                          artifact::Reader& reader, Clock::time_point load_start) {
    if (options.speculative.backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("Engine tp mode has no DFlash speculative target");
    }

    // Graph decode rounds follow the engine option: the TP ranks capture the ordinary and
    // MTP decode schedules per frontier profile with the TpLink spin exchanges (publish,
    // bounded peer spin, consume in one kernel; the go broadcast gates on seq equality,
    // so replays stay correct — core/tp_link.cuh) inside the graph. Prefill stays eager.
    EngineOptions tp_options = options;
    using Package = Qwen3_6_27B;
    const auto& identity                          = reader.identity();
    const Package::WeightsProfile weights_profile = Package::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Package::sampling_defaults(identity.model_id);

    LoadSummary summary;
    summary.target     = identity.model_id == Package::qwen3_8_model_id ? Package::qwen3_8_target_key
                                                                        : Package::target_key;
    summary.model_id   = identity.model_id;
    summary.weights_id = identity.weights_id;

    // Every rank validates and uploads before any rank's artifacts are consumed, so a memory
    // failure names the offending rank and no rank holds half-initialized peers.
    std::array<std::optional<Package::LoadPlan>, 2> plans;
    std::array<std::optional<artifact::MaterializedArtifact>, 2> materialized;
    for (std::uint8_t rank = 0; rank < tp_devices.size(); ++rank) {
        DeviceContext rank_device(tp_devices[rank]);
        artifact::Binder binder(reader);
        plans[rank]         = Package::plan_load(binder, tp_options, weights_profile, rank);
        const std::string rank_label =
            "tp rank " + std::to_string(rank) + " (device " + std::to_string(tp_devices[rank]) + ")";
        const std::size_t preflight_runtime_bytes = runtime_bytes_after_planned_weights(
            plans[rank]->materialization().device_capacity_bytes, rank_label);
        (void)runtime::resolve_kv_capacity(
            tp_options.kv_capacity,
            Package::make_tp_sequence_planner(rank_device, tp_options, weights_profile)
                .capacity_curve(),
            preflight_runtime_bytes);
        auto progress = artifact_progress(tp_options.load_progress);
        materialized[rank].emplace(
            artifact::materialize(reader, plans[rank]->materialization(), rank_device,
                                  progress.callback ? &progress : nullptr));
        rank_device.synchronize();

        const artifact::MaterializationStats stats = materialized[rank]->stats();
        summary.tensor_count += stats.tensor_count;
        summary.resource_count += stats.resource_count;
        summary.artifact_bytes_read += stats.file_bytes;
        summary.host_to_device_bytes += stats.h2d_bytes;
        summary.peak_staging_bytes =
            std::max(summary.peak_staging_bytes, stats.peak_staging_bytes);
        summary.upload_seconds = std::max(summary.upload_seconds, stats.upload_seconds);
    }

    // The last rank's DeviceContext left its device current; restore the primary before any
    // further primary-device allocation, then resolve KV capacity against the tightest rank.
    CUDA_CHECK(cudaSetDevice(primary.device));
    auto model_a = Package::construct_loaded_model(std::move(*plans[0]),
                                                   std::move(*materialized[0]));
    auto model_b = Package::construct_loaded_model(std::move(*plans[1]),
                                                   std::move(*materialized[1]));
    plans[0].reset();
    plans[1].reset();
    primary.synchronize();

    auto sequence_planner = Package::make_tp_sequence_planner(primary, tp_options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    std::size_t min_free_bytes = current_free_device_bytes();
    for (const int device_id : tp_devices) {
        CUDA_CHECK(cudaSetDevice(device_id));
        min_free_bytes = std::min(min_free_bytes, current_free_device_bytes());
    }
    CUDA_CHECK(cudaSetDevice(primary.device));
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(tp_options.kv_capacity, curve, min_free_bytes);
    const std::uint32_t capacity_tokens = capacity_resolution.resolved_tokens;
    Package::TpSequencePlan sequence_plan =
        std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() !=
            capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }

    auto device_a = std::make_unique<DeviceContext>(tp_devices[0]);
    auto device_b = std::make_unique<DeviceContext>(tp_devices[1]);
    auto loaded   = std::make_unique<LoadedQwen3_6_27B>(std::move(model_a));
    auto instance = std::make_unique<Qwen3_6_27BTpInstance>(
        std::move(loaded), std::move(model_b), std::move(device_a), std::move(device_b),
        capacity_resolution, std::move(sequence_plan), primary, options.tp_clock_holder,
        options.tp_clock_holder_hold_ms);
    primary.synchronize();
    std::size_t min_available = current_free_device_bytes();
    for (const int device_id : tp_devices) {
        CUDA_CHECK(cudaSetDevice(device_id));
        min_available = std::min(min_available, current_free_device_bytes());
    }
    CUDA_CHECK(cudaSetDevice(primary.device));
    instance->kv_capacity_resolution.available_after_startup_bytes = min_available;
    summary.load_seconds = std::chrono::duration<double>(Clock::now() - load_start).count();
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

Qwen3_6_27BTpInstance::Qwen3_6_27BTpInstance(
    std::unique_ptr<LoadedQwen3_6_27B> stable_loaded, std::unique_ptr<Qwen3_6_27B::LoadedModel> peer,
    std::unique_ptr<DeviceContext> rank_a, std::unique_ptr<DeviceContext> rank_b,
    runtime::KvCapacityResolution resolution, Qwen3_6_27B::TpSequencePlan sequence_plan,
    DeviceContext& primary, TpClockHolderMode clock_holder, std::uint32_t clock_holder_hold_ms)
    : loaded(std::move(stable_loaded)), peer_model(std::move(peer)), device_a(std::move(rank_a)),
      device_b(std::move(rank_b)), kv_capacity_resolution(resolution),
      request_memory(primary, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_tp_program(*loaded->model, *peer_model, std::move(sequence_plan),
                                             *device_a, *device_b, primary.device,
                                             clock_holder)) {}

Qwen3_6_27BTpInstance::~Qwen3_6_27BTpInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    const bool is_27b_family =
        identity.model_id == Qwen3_6_27B::model_id || identity.model_id == Qwen3_6_27B::qwen3_8_model_id;
    if (options.tp) {
        if (!is_27b_family) {
            throw std::runtime_error("artifact identity '" + identity.model_id + "/" +
                                     identity.weights_id +
                                     "' has no two-rank tensor-parallel target");
        }
        return construct_tp_target_27b(options, device, options.tp_devices, reader, load_start);
    }
    const auto dispatch = [&]<class Target, class Loaded, class Instance>(
                              std::string_view target_key) -> ConstructedTarget {
        return construct_registered<Target, Loaded, Instance>(options, device, reader, load_start,
                                                              target_key);
    };
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return dispatch.operator()<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return dispatch.operator()<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return dispatch.operator()<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
