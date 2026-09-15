#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

// Complete two-rank TP coordinator definition for the executor instantiation in engine.cpp.
// The family impl root is macro-parameterized per exact Variant; these two includes are the
// only macro-gated includes in the composition root and stay local to this header.
#include "targets/qwen3_6_27b/impl/tp_variant.h"
#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::TpVariant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_tp_runtime
#include "targets/qwen3_6/impl/runtime/tp_program.h"
#undef NINFER_QWEN36_VARIANT
#undef NINFER_QWEN36_RUNTIME_NS

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    explicit LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

// The ConcurrentExecutor's package surface for the two-rank tensor-parallel instance: the
// coordinator plays Program, and the plans are the shard Variant's.
struct Qwen3_6_27BTpPackage {
    using Program         = Qwen3_6_27B::TpProgram;
    using RequestBasePlan = Qwen3_6_27B::TpRequestBasePlan;
    using RequestPlan     = Qwen3_6_27B::TpRequestPlan;
};

// One engine running one model tensor-parallel across both tp_devices: per-rank loaded shard
// and DeviceContext, and the TpProgram coordinator as the executor's Program. Token
// publication, the frontend, and request transients stay single-copy on the primary device.
struct Qwen3_6_27BTpInstance {
    using Package = Qwen3_6_27BTpPackage;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    std::unique_ptr<Qwen3_6_27B::LoadedModel> peer_model;
    std::unique_ptr<DeviceContext> device_a;
    std::unique_ptr<DeviceContext> device_b;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::TpProgram> program;

    Qwen3_6_27BTpInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                          std::unique_ptr<Qwen3_6_27B::LoadedModel> peer,
                          std::unique_ptr<DeviceContext> rank_a, std::unique_ptr<DeviceContext> rank_b,
                          runtime::KvCapacityResolution resolution,
                          Qwen3_6_27B::TpSequencePlan sequence_plan, DeviceContext& primary,
                          TpClockHolderMode clock_holder = TpClockHolderMode::Demand,
                          std::uint32_t clock_holder_hold_ms = 10'000);
    ~Qwen3_6_27BTpInstance();

    Qwen3_6_27BTpInstance(const Qwen3_6_27BTpInstance&)            = delete;
    Qwen3_6_27BTpInstance& operator=(const Qwen3_6_27BTpInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    explicit LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BInstance>,
                                  std::unique_ptr<Qwen3_6_35BA3BInstance>,
                                  std::unique_ptr<Qwen3_6_27BTpInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
