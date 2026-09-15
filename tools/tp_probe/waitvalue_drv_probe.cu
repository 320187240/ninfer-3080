// waitvalue_drv_probe — Q12: can the CUDA DRIVER API's cuStreamWaitValue64 gate a cross-GPU
// exchange on this WDDM pair? The runtime API surface was removed in CUDA 13, but the driver
// entry point remains. The wait is evaluated by the GPU channel scheduler (not an SM), so it
// may be a reliable observer of the peer's mapped-pinned release writes where an SM
// ld.global.cv spin is not. Probes: (a) entry-point availability, (b) a 2000-exchange chained
// publish→wait→consume loop on both devices with payload markers, (c) the engine's
// enqueue-ahead shape — one host thread enqueues side0's whole 130-exchange round, then
// side1's, with the waits stream-ordered (no host callbacks), verifying no queue-full
// deadlock, and (d) large 21 MB payloads matching the prefill exchange width.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using bf16 = __nv_bfloat16;
using u64  = unsigned long long;
using u32  = unsigned int;

static const int  HIDDEN = 5120;
static const int  BLOCK  = 256;
static const int  MAX_BLOCKS = 64;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %d: %s\n", cudaGetErrorName(e_), __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char* s_ = nullptr; cuGetErrorString(r_, &s_); \
    std::fprintf(stderr, "CU error at %d: %s\n", __LINE__, s_ ? s_ : "?"); std::exit(1); } } while (0)

struct TpLinkControl {
    volatile u64 release[2][2];
    volatile u64 error[2];
};

struct Slot { u32 arrive; };

__device__ __forceinline__ void tp_store(volatile u64* a, u64 v) {
    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned short tp_load(const volatile bf16* a) {
    unsigned short v = 0; asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(v) : "l"(a)); return v;
}
__device__ __forceinline__ void tp_store(volatile bf16* a, bf16 v) {
    const unsigned short b = __bfloat16_as_ushort(v);
    asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(a), "h"(b) : "memory");
}
__device__ __forceinline__ bf16 tp_bf16(unsigned short b) { return __ushort_as_bfloat16(b); }

struct Sync { TpLinkControl* control; int side; int parity; u64 seq; Slot* slot; };

__global__ void publish_add_kernel(bf16* local, size_t n, volatile bf16* my_payload, Sync s) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        tp_store(my_payload + i, local[i]);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const u32 arrived = atomicAdd(&s.slot->arrive, 1u);
        if (arrived + 1u == gridDim.x) {
            tp_store(&s.control->release[s.side][s.parity], s.seq);
            s.slot->arrive = 0u;
        }
    }
}

__global__ void consume_add_kernel(bf16* local, size_t n, volatile const bf16* peer_payload,
                                   Sync s, u64* bad) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local[i] = __float2bfloat16(__bfloat162float(local[i]) +
                                    __bfloat162float(tp_bf16(tp_load(peer_payload + i))));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned short lo = tp_load(peer_payload);
        const unsigned short hi = tp_load(peer_payload + 1);
        const u64 pseq = static_cast<u64>(__bfloat162float(tp_bf16(lo))) +
                         static_cast<u64>(__bfloat162float(tp_bf16(hi))) * 256ULL;
        if (pseq != s.seq) { *bad = s.seq * 1000000ULL + pseq; }
    }
}

__global__ void fill_marker(bf16* p, size_t n, u64 seq) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float v = static_cast<float>(i & 1023) * 0.001F;
        if (i == 0) { v = static_cast<float>(seq & 0xffu); }
        if (i == 1) { v = static_cast<float>((seq >> 8) & 0xffffu); }
        p[i] = __float2bfloat16(v);
    }
}

__global__ void compute_kernel(float* state, int steps, size_t dim) {
    const size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (base >= dim) { return; }
    float x = state[base];
    for (int t = 0; t < steps; ++t) { x = x * 0.9999F + 0.0001F * __sinf(x + t); }
    state[base] = x;
}

static char* map_host = nullptr;
static char* map_dev[2] = {};
static TpLinkControl* ctrl = nullptr;
static size_t stride = 0;

struct Side {
    int dev; cudaStream_t stream; cudaEvent_t ev;
    bf16* local; float* cstate; u64* bad; Slot* slot; u64 seq;
} S[2];

static volatile bf16* payload(int side, int par) {
    return reinterpret_cast<volatile bf16*>(map_dev[side] + static_cast<size_t>(side * 2 + par) * stride);
}
static Sync sync_for(int side, u64 seq) {
    return Sync{reinterpret_cast<TpLinkControl*>(map_dev[side] + stride * 4), side,
                static_cast<int>(seq & 1u), seq, S[side].slot};
}
static unsigned int blocks_for(size_t n) {
    size_t b = (n + BLOCK - 1) / BLOCK; if (b < 1) b = 1; if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return static_cast<unsigned int>(b);
}

typedef CUresult (*PFN_cuswv64)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
static PFN_cuswv64 g_wait64 = nullptr;
static const unsigned int kGeq = 0x0; // CU_STREAM_WAIT_VALUE_GEQ

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 16;
    const size_t n_ex = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : HIDDEN;

    int count = 0;
    CK(cudaGetDeviceCount(&count));
    if (count < 2) { std::fprintf(stderr, "need two devices\n"); return 2; }

    void* entry = nullptr;
    const cudaError_t epr = cudaGetDriverEntryPoint("cuStreamWaitValue64", &entry,
                                                    cudaEnableDefault, nullptr);
    g_wait64 = reinterpret_cast<PFN_cuswv64>(entry);
    std::printf("cudaGetDriverEntryPoint(cuStreamWaitValue64): %s\n",
                epr == cudaSuccess ? "OK" : cudaGetErrorName(epr));
    if (epr != cudaSuccess || g_wait64 == nullptr) {
        std::printf("RESULT: WAITVALUE_DRV_UNAVAILABLE\n");
        return 3;
    }

    const size_t max_elems = static_cast<size_t>(2048) * HIDDEN;
    stride = ((max_elems * 2 + 63) / 64) * 64;
    const size_t total = stride * 4 + sizeof(TpLinkControl);
    CK(cudaHostAlloc(reinterpret_cast<void**>(&map_host), total,
                     cudaHostAllocMapped | cudaHostAllocPortable));
    ctrl = reinterpret_cast<TpLinkControl*>(map_host + stride * 4);
    for (int side = 0; side < 2; ++side) {
        S[side].dev = side;
        CK(cudaSetDevice(side));
        CK(cudaStreamCreate(&S[side].stream));
        CK(cudaEventCreateWithFlags(&S[side].ev, cudaEventDisableTiming));
        CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&map_dev[side]), map_host, 0));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].local), max_elems * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].cstate), static_cast<size_t>(HIDDEN) * 4 * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].bad), 8));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].slot), sizeof(Slot) * 2));
        CK(cudaMemset(S[side].cstate, 0, static_cast<size_t>(HIDDEN) * 4 * 4));
        CK(cudaMemset(S[side].slot, 0, sizeof(Slot) * 2));
        CK(cudaMemset(S[side].bad, 0, 8));
    }
    // The wait address each side polls: the PEER's release word, via THIS side's device
    // mapping of the shared control block.
    TpLinkControl* ctrl_dev[2] = {
        reinterpret_cast<TpLinkControl*>(map_dev[0] + stride * 4),
        reinterpret_cast<TpLinkControl*>(map_dev[1] + stride * 4)};

    auto exchange_add = [&](int side) {
        Side& s = S[side];
        const u64 seq = ++s.seq;
        const int par = static_cast<int>(seq & 1u);
        const Sync sy = sync_for(side, seq);
        CK(cudaSetDevice(s.dev));
        fill_marker<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(s.local, n_ex, seq);
        publish_add_kernel<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(s.local, n_ex, payload(side, par), sy);
        CKD(g_wait64((CUstream)s.stream,
                     reinterpret_cast<CUdeviceptr>(
                         const_cast<unsigned long long*>(&ctrl_dev[side]->release[1 - side][par])),
                     seq, kGeq));
        consume_add_kernel<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(
            s.local, n_ex, payload(1 - side, par), sy, s.bad);
    };

    int first_bad = -1;
    for (int r = 0; r < rounds && first_bad < 0; ++r) {
        S[0].seq = S[1].seq = 0;
        std::memset(map_host, 0, total);
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(S[side].dev));
            CK(cudaMemsetAsync(S[side].slot, 0, sizeof(Slot) * 2, S[side].stream));
            CK(cudaMemsetAsync(S[side].bad, 0, 8, S[side].stream));
        }
        // Engine shape: side0's whole round enqueued first, then side1's, one host thread.
        auto enqueue_round = [&](int side) {
            exchange_add(side);
            for (int layer = 0; layer < 64; ++layer) {
                compute_kernel<<<48, 256, 0, S[side].stream>>>(
                    S[side].cstate, 20000, static_cast<size_t>(HIDDEN) * 4);
                exchange_add(side);
                compute_kernel<<<48, 256, 0, S[side].stream>>>(
                    S[side].cstate, 20000, static_cast<size_t>(HIDDEN) * 4);
                exchange_add(side);
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        enqueue_round(0);
        for (int side = 0; side < 2; ++side) { // lead_sync_ dual flush
            CK(cudaSetDevice(S[side].dev));
            CK(cudaEventRecord(S[side].ev, S[side].stream));
            (void)cudaEventQuery(S[side].ev);
        }
        enqueue_round(1);
        bool pa = true, pb = true;
        while (pa || pb) {
            if (pa) {
                CK(cudaSetDevice(0));
                const cudaError_t st = cudaStreamQuery(S[0].stream);
                if (st != cudaErrorNotReady) { CK(st); pa = false; }
            }
            if (pb) {
                CK(cudaSetDevice(1));
                const cudaError_t st = cudaStreamQuery(S[1].stream);
                if (st != cudaErrorNotReady) { CK(st); pb = false; }
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        u64 bad[2] = {0, 0};
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(side));
            CK(cudaMemcpy(bad, S[side].bad, 8, cudaMemcpyDeviceToHost));
        }
        std::printf("round %d: %.1f ms (%.0f us/exch n=%zu), seq=%llu/%llu, "
                    "rel=%llu/%llu/%llu/%llu, bad=%llu/%llu\n",
                    r, ms, ms * 1000.0 / 130.0, n_ex, S[0].seq, S[1].seq,
                    ctrl->release[0][0], ctrl->release[0][1], ctrl->release[1][0],
                    ctrl->release[1][1], bad[0], bad[1]);
        std::fflush(stdout);
        if (bad[0] || bad[1] || ctrl->error[0] || ctrl->error[1]) { first_bad = r; }
    }
    std::printf(first_bad < 0 ? "RESULT: WAITVALUE_DRV_CLEAN rounds=%d\n"
                              : "RESULT: WAITVALUE_DRV_BAD round=%d\n",
                first_bad < 0 ? rounds : first_bad);
    return first_bad < 0 ? 0 : 1;
}
