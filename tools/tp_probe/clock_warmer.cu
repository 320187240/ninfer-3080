// clock_warmer — diagnostic: keeps one small grid resident so WDDM's clock governor
// sees continuous utilization and boosts SM clocks. Not part of the engine.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void warm_kernel(volatile int* stop) {
    float x = 1.0F;
    while (*stop == 0) {
        #pragma unroll 8
        for (int i = 0; i < 64; ++i) { x = x * 0.9999F + 0.0001F; }
        __nanosleep(2000);
    }
    if (x == 42.0F) { printf(""); }
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    int* stop = nullptr;
    cudaSetDevice(dev);
    cudaMalloc(&stop, 4);
    cudaMemset(stop, 0, 4);
    warm_kernel<<<1, 64>>>(stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("warmer launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf("warmer running on device %d; press enter to stop\n", dev);
    std::fflush(stdout);
    std::getchar();
    cudaMemset(stop, 1, 4);
    cudaDeviceSynchronize();
    cudaFree(stop);
    return 0;
}
