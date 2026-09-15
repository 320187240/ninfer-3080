// waitvalue_graph_probe — Q13: can the StreamWait exchange protocol be captured into
// per-rank CUDA graphs and replayed correctly? Tests the exact M4a design:
//   publish kernel (payload wt-stores + release word = code)
//   -> cuStreamWaitValue64(peer release word, code, GEQ)   [captured?]
//   -> consume kernel (peer payload cv-loads + reset of the gated word to 0)
// The consume-side reset is the replay re-arm: every exchange returns its flag word to
// zero, so a replay's GEQ(code) gate can only pass on this replay's publish. Codes are
// baked per node at capture time (unique per exchange index), and the reset discipline
// keeps the word at 0 between exchanges, so eager and captured exchanges can interleave
// on the same slots. Verification is exact integer math: every consume adds the peer
// payload's bf16-exact round tag into a device accumulator, so 130 exchanges per replay
// must sum to 130*tag — any stale consumption of a previous replay's payload shows up
// as a wrong sum. Transport words must also return to zero after every round without any
// host reset, proving the re-arm.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using bf16 = __nv_bfloat16;
using u64  = unsigned long long;
using u32  = unsigned int;

static const int HIDDEN        = 5120;
static const int BLOCK         = 256;
static const int MAX_BLOCKS    = 64;
static const int EXCHANGES     = 130; // 1 embedding + 64 layers x 2 + 1 argmax
static const int COMPUTE_STEPS = 1800;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %d: %s\n", cudaGetErrorName(e_), __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

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

__device__ __forceinline__ void publish_tail(Sync s) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const u32 arrived = atomicAdd(&s.slot->arrive, 1u);
        if (arrived + 1u == gridDim.x) {
            tp_store(&s.control->release[s.side][s.parity], s.seq);
            s.slot->arrive = 0u;
        }
    }
}

__global__ void publish_add_kernel(bf16* local, size_t n, volatile bf16* my_payload, Sync s) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        tp_store(my_payload + i, local[i]);
    }
    __threadfence_system();
    publish_tail(s);
}

// Consume with the replay re-arm: after the payload loads, block 0 resets the gated peer
// word to zero so the next use of the slot must wait for a fresh publish, and folds the
// peer payload's round tag into the exact-integer accumulator. The exchange math is a
// copy (not an add) so payload element 0 carries the pristine fill tag every exchange —
// this probe tests the transport, not the allreduce arithmetic.
__global__ void consume_add_kernel(bf16* local, size_t n, volatile const bf16* peer_payload,
                                   volatile u64* peer_release, u32* acc) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        local[i] = tp_bf16(tp_load(peer_payload + i));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned short lo = tp_load(peer_payload);
        const u32 tag = static_cast<u32>(__bfloat162float(tp_bf16(lo)));
        atomicAdd(acc, tag);
        tp_store(peer_release, 0ULL); // re-arm: word returns to zero for the next use
    }
}

// Round-tagged payload fill; runs as the round's first kernel, reading the round tag the
// host staged into device memory just before the replay.
__global__ void fill_tagged(bf16* p, size_t n, const u32* tag_in) {
    const u32 tag = *tag_in;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float v = static_cast<float>(i & 1023) * 0.001F;
        if (i == 0) { v = static_cast<float>(tag); }
        if (i == 1) { v = 0.0F; }
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
static bool g_capturing = false;
static int g_wait_capture_err = 0; // first nonzero CUresult from the wait during capture

struct Side {
    int dev; cudaStream_t stream;
    bf16* local; float* cstate; Slot* slot; u32* tag_dev; u32* acc;
    u64 seq;               // per-side exchange counter; codes are baked from it
    cudaGraphExec_t graph = nullptr;
} S[2];

static volatile bf16* payload(int side, int par) {
    return reinterpret_cast<volatile bf16*>(map_dev[side] + static_cast<size_t>(side * 2 + par) * stride);
}
static Sync sync_for(int side, u64 seq) {
    return Sync{reinterpret_cast<TpLinkControl*>(map_dev[side] + stride * 4), side,
                static_cast<int>(seq & 1u), seq, S[side].slot};
}
static TpLinkControl* ctrl_dev(int side) {
    return reinterpret_cast<TpLinkControl*>(map_dev[side] + stride * 4);
}
static unsigned int blocks_for(size_t n) {
    size_t b = (n + BLOCK - 1) / BLOCK; if (b < 1) b = 1; if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return static_cast<unsigned int>(b);
}

typedef CUresult (*PFN_cuswv64)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
static PFN_cuswv64 g_wait64 = nullptr;
static const unsigned int kGeq = 0x0; // CU_STREAM_WAIT_VALUE_GEQ

static void wait_peer(int side, int par, u64 seq) {
    const CUresult r = g_wait64((CUstream)S[side].stream,
                                reinterpret_cast<CUdeviceptr>(&ctrl_dev(side)->release[1 - side][par]),
                                seq, kGeq);
    if (g_capturing) {
        if (r != CUDA_SUCCESS && g_wait_capture_err == 0) { g_wait_capture_err = static_cast<int>(r); }
        return; // capture verdict comes from cudaStreamEndCapture
    }
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(r, &s);
        std::fprintf(stderr, "cuStreamWaitValue64 failed at seq %llu: %s\n",
                     static_cast<unsigned long long>(seq), s ? s : "?");
        std::exit(1);
    }
}

static const size_t n_ex = HIDDEN;

// One side's 130-exchange round body, used for both eager enqueue and graph capture. The
// exchange code is ++seq at each call; during capture these are baked per node.
static void enqueue_round_side(int side) {
    Side& s = S[side];
    CK(cudaSetDevice(s.dev));
    fill_tagged<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(s.local, n_ex, s.tag_dev);
    for (int e = 0; e < EXCHANGES; ++e) {
        const u64 seq = ++s.seq;
        const int par = static_cast<int>(seq & 1u);
        const Sync sy = sync_for(side, seq);
        publish_add_kernel<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(s.local, n_ex,
                                                                     payload(side, par), sy);
        wait_peer(side, par, seq);
        consume_add_kernel<<<blocks_for(n_ex), BLOCK, 0, s.stream>>>(
            s.local, n_ex, payload(1 - side, par), &ctrl_dev(side)->release[1 - side][par], s.acc);
        if (true) {
            compute_kernel<<<48, 256, 0, s.stream>>>(s.cstate, COMPUTE_STEPS,
                                                     static_cast<size_t>(HIDDEN) * 4);
        }
    }
}

static std::atomic<long long> g_round_deadline_ms{0};

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 100;

    // Watchdog: a wedged gate would hang the probe; dump transport state and exit.
    std::thread watchdog([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const long long deadline = g_round_deadline_ms.load();
            if (deadline == 0) { continue; }
            const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - deadline > 60000) {
                std::printf("WATCHDOG: round exceeded 60 s; release a0/a1/b0/b1="
                            "%llu/%llu/%llu/%llu\n",
                            static_cast<unsigned long long>(ctrl->release[0][0]),
                            static_cast<unsigned long long>(ctrl->release[0][1]),
                            static_cast<unsigned long long>(ctrl->release[1][0]),
                            static_cast<unsigned long long>(ctrl->release[1][1]));
                std::fflush(stdout);
                std::_Exit(9);
            }
        }
    });
    watchdog.detach();

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
    std::memset(map_host, 0, total);
    for (int side = 0; side < 2; ++side) {
        S[side].dev = side;
        CK(cudaSetDevice(side));
        CK(cudaStreamCreate(&S[side].stream));
        CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&map_dev[side]), map_host, 0));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].local), max_elems * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].cstate), static_cast<size_t>(HIDDEN) * 4 * 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].slot), sizeof(Slot) * 2));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].tag_dev), 4));
        CK(cudaMalloc(reinterpret_cast<void**>(&S[side].acc), 4));
        CK(cudaMemset(S[side].cstate, 0, static_cast<size_t>(HIDDEN) * 4 * 4));
        CK(cudaMemset(S[side].slot, 0, sizeof(Slot) * 2));
        CK(cudaMemset(S[side].acc, 0, 4));
        S[side].seq = 0;
    }

    auto sync_both = [&] {
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(side));
            CK(cudaStreamSynchronize(S[side].stream));
        }
    };

    // ---- eager baseline (the M3 shape) -------------------------------------------
    double eager_ms = 0.0;
    for (int r = 0; r < 8; ++r) {
        for (int side = 0; side < 2; ++side) {
            S[side].seq = 0;
            CK(cudaSetDevice(side));
            CK(cudaMemsetAsync(S[side].acc, 0, 4, S[side].stream));
            const u32 tag = static_cast<u32>(r + 1);
            CK(cudaMemcpyAsync(S[side].tag_dev, &tag, 4, cudaMemcpyHostToDevice, S[side].stream));
        }
        sync_both();
        std::memset(map_host, 0, total); // words zero with both ranks idle
        const auto t0  = std::chrono::steady_clock::now();
        std::thread follow([&] { enqueue_round_side(1); });
        enqueue_round_side(0);
        follow.join();
        sync_both();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        if (r >= 4) { eager_ms += ms; }
        u32 acc[2] = {0, 0};
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(side));
            CK(cudaMemcpy(acc + side, S[side].acc, 4, cudaMemcpyDeviceToHost));
        }
        const u32 tag    = static_cast<u32>(r + 1);
        const u32 expect = static_cast<u32>(EXCHANGES) * tag;
        if (acc[0] != expect || acc[1] != expect) {
            std::printf("RESULT: EAGER_BAD round=%d acc=%u/%u expect=%u\n", r, acc[0], acc[1], expect);
            return 1;
        }
    }
    std::printf("eager round (%d exchanges): %.2f ms avg over 4\n", EXCHANGES, eager_ms / 4.0);

    // ---- capture ------------------------------------------------------------------
    for (int side = 0; side < 2; ++side) { S[side].seq = 0; }
    std::memset(map_host, 0, total);
    sync_both();
    cudaGraph_t graphs[2] = {};
    for (int side = 0; side < 2; ++side) {
        CK(cudaSetDevice(side));
        CK(cudaMemsetAsync(S[side].acc, 0, 4, S[side].stream));
        u32 tag0 = 1;
        CK(cudaMemcpyAsync(S[side].tag_dev, &tag0, 4, cudaMemcpyHostToDevice, S[side].stream));
        CK(cudaStreamSynchronize(S[side].stream));
        g_capturing        = true;
        g_wait_capture_err = 0;
        CK(cudaStreamBeginCapture(S[side].stream, cudaStreamCaptureModeThreadLocal));
        enqueue_round_side(side); // errors inside surface at EndCapture
        cudaError_t cap_err = cudaStreamEndCapture(S[side].stream, &graphs[side]);
        g_capturing = false;
        if (cap_err != cudaSuccess) {
            std::printf("RESULT: CAPTURE_FAILED side=%d %s (wait CUresult during capture=%d)\n",
                        side, cudaGetErrorName(cap_err), g_wait_capture_err);
            return 4;
        }
        if (g_wait_capture_err != 0) {
            std::printf("RESULT: WAIT_DURING_CAPTURE_REJECTED side=%d CUresult=%d\n", side,
                        g_wait_capture_err);
            return 5;
        }
        CK(cudaGraphInstantiate(&S[side].graph, graphs[side], 0));
        CK(cudaGraphUpload(S[side].graph, S[side].stream));
        CK(cudaStreamSynchronize(S[side].stream));
        std::printf("side %d graph captured+instantiated (%llu exchanges baked, codes 1..%llu)\n",
                    side, static_cast<unsigned long long>(S[side].seq),
                    static_cast<unsigned long long>(S[side].seq));
    }
    if (S[0].seq != S[1].seq) {
        std::printf("RESULT: CODE_MISMATCH %llu vs %llu\n",
                    static_cast<unsigned long long>(S[0].seq),
                    static_cast<unsigned long long>(S[1].seq));
        return 6;
    }

    // ---- replay correctness + timing ----------------------------------------------
    // No transport reset inside the loop: the words must return to zero through the
    // consume kernels' own re-arm, which the per-round checks verify.
    int first_bad = -1;
    double graph_ms = 0.0;
    for (int r = 0; r < rounds && first_bad < 0; ++r) {
        const u32 tag = static_cast<u32>(r + 1);
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(side));
            CK(cudaMemsetAsync(S[side].acc, 0, 4, S[side].stream));
            CK(cudaMemcpyAsync(S[side].tag_dev, &tag, 4, cudaMemcpyHostToDevice, S[side].stream));
        }
        sync_both();
        g_round_deadline_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() + 60000);
        const auto t0 = std::chrono::steady_clock::now();
        std::thread follow([&] {
            CK(cudaSetDevice(1));
            CK(cudaGraphLaunch(S[1].graph, S[1].stream));
        });
        CK(cudaSetDevice(0));
        CK(cudaGraphLaunch(S[0].graph, S[0].stream));
        follow.join();
        sync_both();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        g_round_deadline_ms.store(0);
        if (r >= 4) { graph_ms += ms; }
        u32 acc[2] = {0, 0};
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(side));
            CK(cudaMemcpy(acc + side, S[side].acc, 4, cudaMemcpyDeviceToHost));
        }
        const u32 expect = static_cast<u32>(EXCHANGES) * tag;
        const bool armed = ctrl->release[0][0] == 0 && ctrl->release[0][1] == 0 &&
                           ctrl->release[1][0] == 0 && ctrl->release[1][1] == 0;
        if (acc[0] != expect || acc[1] != expect || !armed) {
            first_bad = r;
            std::printf("round %d BAD: acc=%u/%u expect=%u rel=%llu/%llu/%llu/%llu\n", r, acc[0],
                        acc[1], expect, static_cast<unsigned long long>(ctrl->release[0][0]),
                        static_cast<unsigned long long>(ctrl->release[0][1]),
                        static_cast<unsigned long long>(ctrl->release[1][0]),
                        static_cast<unsigned long long>(ctrl->release[1][1]));
        } else if (r < 3 || (r % 25) == 0) {
            std::printf("round %d ok: %.2f ms\n", r, ms);
        }
        std::fflush(stdout);
    }
    std::printf("graph round (%d exchanges): %.2f ms avg over %d\n", EXCHANGES, graph_ms / (rounds - 4.0),
                rounds - 4);
    std::printf(first_bad < 0
                    ? "RESULT: WAITVALUE_GRAPH_CLEAN rounds=%d eager=%.2fms graph=%.2fms\n"
                    : "RESULT: WAITVALUE_GRAPH_BAD round=%d\n",
                first_bad < 0 ? rounds : first_bad, eager_ms / 4.0, graph_ms / (rounds - 4.0));
    return first_bad < 0 ? 0 : 1;
}
