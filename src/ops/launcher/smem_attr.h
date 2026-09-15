#pragma once

// Host-side helpers shared by Op launchers.

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <unordered_map>

namespace ninfer::ops::detail {

// Opt a kernel into more than the 48 KiB default dynamic shared memory. Function attributes
// are per-device module state, so a process-static cache would under-serve every device after
// the first; this memoizes per (kernel, current device) instead.
inline void set_max_dynamic_smem_per_device(const void* kernel, std::size_t bytes) {
    static thread_local std::unordered_map<const void*, std::array<std::uint8_t, 32>> done;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    auto& per_device = done[kernel];
    if (device >= 0 && static_cast<std::size_t>(device) < per_device.size()) {
        const auto index = static_cast<std::size_t>(device);
        if (per_device[index] != 0) { return; }
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes)));
        per_device[index] = 1;
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(bytes)));
}

template <class Kernel>
inline void set_max_dynamic_smem_per_device(Kernel kernel, std::size_t bytes) {
    set_max_dynamic_smem_per_device(reinterpret_cast<const void*>(kernel), bytes);
}

} // namespace ninfer::ops::detail
