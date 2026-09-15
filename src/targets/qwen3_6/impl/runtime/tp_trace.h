#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.
//
// Debug-only layer-granularity parity trace for the two-rank tensor-parallel port. When
// NINFER_TP_TRACE names a directory, every prefill capture point below copies its device
// tensor to the host (stream-ordered, then synced) and writes one raw BF16 file
// "<dir>/<rank>_<stage>_D<ne0>_T<ne1>.bf16" (rank is "sg", "tp0", or "tp1"). The dump is
// inert without the environment variable, so production runs pay one getenv probe.

#include "core/tensor.h"
#include "targets/qwen3_6/impl/runtime/tp_exec.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

[[nodiscard]] inline const char* tp_trace_dir() {
    static const char* dir = [] {
        const char* value = std::getenv("NINFER_TP_TRACE");
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }();
    return dir;
}

// rank label: "sg" on the single-GPU schedule, "tp0"/"tp1" per TpExec side.
[[nodiscard]] inline const char* tp_trace_rank(const TpExec* tp) {
    return tp == nullptr ? "sg" : (tp->side == 0 ? "tp0" : "tp1");
}

inline void tp_trace_dump(const char* rank, const char* stage, const Tensor& x,
                          cudaStream_t stream) {
    const char* dir = tp_trace_dir();
    if (dir == nullptr) { return; }
    if (x.data == nullptr || x.ne[0] <= 0 || x.ne[1] <= 0 ||
        (x.dtype != DType::BF16 && x.dtype != DType::FP32)) {
        return;
    }
    std::vector<char> host(x.bytes());
    if (cudaMemcpyAsync(host.data(), x.data, x.bytes(), cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s_%s_D%lld_T%lld.%s", dir, rank, stage,
                  static_cast<long long>(x.ne[0]), static_cast<long long>(x.ne[1]),
                  x.dtype == DType::FP32 ? "f32" : "bf16");
    if (std::FILE* file = std::fopen(path, "wb")) {
        (void)std::fwrite(host.data(), 1, host.size(), file);
        (void)std::fclose(file);
    }
}

// One scalar trace line ("<key>=<value>") appended to <dir>/<rank>_scalars.txt; used for the
// final prefill argmax winner per rank.
inline void tp_trace_scalar(const char* rank, const char* key, long long value) {
    const char* dir = tp_trace_dir();
    if (dir == nullptr) { return; }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s_scalars.txt", dir, rank);
    if (std::FILE* file = std::fopen(path, "a")) {
        (void)std::fprintf(file, "%s=%lld\n", key, value);
        (void)std::fclose(file);
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
