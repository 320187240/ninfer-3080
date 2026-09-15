// tp_engine_repro — standalone reproduction of the TP2 engine's mapped-pinned spin-handshake
// stall. Mirrors the engine execution shape exactly (src/core/tp_link.{cu,cuh} protocol copied
// verbatim + instrumentation, src/targets/qwen3_6/impl/runtime/tp_program.h host pattern):
//
//   - ONE worker thread alternates devices: side 0's whole round is enqueued first, flushed
//     (event record + query), then side 1's whole mirrored round, flushed, then both streams
//     are polled to completion (cudaStreamQuery loop) — the lead_sync_ pattern.
//   - ~130 chained parity-double-buffered exchanges per round (1 embedding + 2 x 64 layers +
//     1 argmax) interleaved with representative compute: large bf16 GEMMs on every layer, a
//     long sequential FP32 GDN-recurrence-like kernel on GDN layers, and a KV-scan attention
//     kernel on full-attention layers (layers 3, 7, 11, ... like kHybridAttentionInterval=4).
//   - Per-exchange device-side instrumentation: the payload's first two elements carry the
//     publisher's seq; the consumer records which seq it actually consumed plus the peer
//     release value it observed, so a stale payload after a spin timeout is directly visible.
//
// First stall logs (side, exchange index, seq, parity, release matrix, consumed-vs-expected
// payload seq). Switches allow minimizing the trigger (drop kernel classes, single thread,
// shrink sizes). See tools/tp_probe/README.md for the probe family this extends.

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

// ---------------- engine-mirrored constants (qwen3.6-27b TP shard) ----------------
static const int  HIDDEN       = 5120;    // full hidden width
static const int  LAYERS       = 64;
static const int  INTER_SHARD  = 8704;    // 17408 / 2 shard intermediate
static const int  VOCAB_ROWS   = 124160;  // 248320 / 2 shard rows
static const int  ATT_INTERVAL = 4;       // (layer+1)%4==0 -> full attention
static const int  BLOCK        = 256;
static const int  MAX_BLOCKS   = 64;      // engine residency cap (68-SM peer)

static const u64  SPIN_BUDGET_DEFAULT = 2500000000ULL; // engine kTpLinkSpinCycles (~1 s @ 2.5 GHz)

static u64 g_spin_budget = SPIN_BUDGET_DEFAULT;
static int  g_rounds     = 8;
static int  g_mode       = 0;      // 0 = decode (1 token), 1 = prefill (2048-token chunk)
static bool g_gemm       = true;
static bool g_gdn        = true;
static bool g_attn       = true;
static int  g_threads    = 1;      // 1 = engine pattern (one worker thread alternating)
static int  g_tokens     = 1;      // decode default; prefill mode sets 2048
static int  g_max_exch_log = 4096;
static int  g_gemm_k       = 5120;  // k-dim cap for the GEMM stand-in (calibrates per-gap cost)
static int  g_gemm_blocks = 0;     // >0 caps the GEMM grid's x dim (per-gap cost calibration)
static int  g_gate        = 0;     // 0 = spin kernel, 1 = stream-wait (publish/wait/consume)
static int  g_spin_mode   = 1;     // 0 = relayed go broadcast (pre-M4c), 1 = every-block spin (M4c)
static int  g_graph       = 0;     // 1 = capture each side's round into a CUDA graph, replay
static bool g_capturing   = false; // a ThreadLocal capture is active: cudaSetDevice would invalidate it

typedef CUresult (*wait64_fn)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
static wait64_fn g_wait64 = nullptr;
static const unsigned int kGeq = 0x0; // CU_STREAM_WAIT_VALUE_GEQ

static int  g_dev[2]  = {0, 1};
static const char* g_name[2] = {"dev0", "dev1"};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// ---------------- transport (copied from src/core/tp_link.cuh) ----------------
struct TpLinkControl {
    volatile u64 release[2][2]; // [side][parity] = seq once that side published
    volatile u64 error[2];      // [side] nonzero after a bounded-spin timeout
};

struct TpLinkDeviceSlot {
    volatile u64 go;   // spin_mode 0 only: block 0's go broadcast
    u32 arrive;        // grid-wide arrival counter; the last block resets it to zero
    u32 rearm;         // spin_mode 1 only: grid-wide post-gate counter for the re-arm
};

struct TpLinkSync {
    TpLinkControl* control;
    int side;
    int parity;
    u64 seq;
    TpLinkDeviceSlot* slot;
};

__device__ __forceinline__ u64 tp_load(const volatile u64* address) {
    u64 value = 0;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(value) : "l"(address));
    return value;
}

__device__ __forceinline__ void tp_store(volatile u64* address, u64 value) {
    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ unsigned short tp_load(const volatile bf16* address) {
    unsigned short value = 0;
    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(value) : "l"(address));
    return value;
}

__device__ __forceinline__ void tp_store(volatile bf16* address, bf16 value) {
    const unsigned short bits = __bfloat16_as_ushort(value);
    asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(address), "h"(bits) : "memory");
}

__device__ __forceinline__ bf16 tp_bf16(unsigned short bits) {
    return __ushort_as_bfloat16(bits);
}

// Per-exchange instrumentation record, written by block 0 thread 0 of each side.
struct ExchLog {
    u64 seq;                 // this side's exchange seq
    u64 peer_release_seen;   // peer release value at consume time (after spin)
    u64 consumed_payload_seq; // seq marker decoded from the peer payload actually consumed
    u64 spin_cycles;         // block-0 peer-flag wait cost in clock64 cycles
    u64 timeout;             // 0 = clean, 1 = peer-flag spin timed out, 2 = go-broadcast timed out
    u64 own_release;         // this side's release value after the exchange
    u64 reserved;
};

__device__ __forceinline__ void tp_arrive_and_wait(TpLinkSync sync, u64 spin_budget,
                                                   ExchLog* log, int spin_mode) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const u32 arrived = atomicAdd(&sync.slot->arrive, 1u);
        if (arrived + 1u == gridDim.x) {
            tp_store(&sync.control->release[sync.side][sync.parity], sync.seq); // release
            sync.slot->arrive = 0u;
        }
    }
    if (spin_mode == 0) {
        // Pre-M4c engine design: block 0 bounded-spins on the peer release, then broadcasts
        // a device-local go flag the rest of the grid spins on.
        if (blockIdx.x == 0) {
            if (threadIdx.x == 0) {
                const u64 start = clock64();
                u64 timeout     = 0;
                while (tp_load(&sync.control->release[1 - sync.side][sync.parity]) < sync.seq) {
                    if (static_cast<u64>(clock64() - start) > spin_budget) {
                        tp_store(&sync.control->error[sync.side], sync.seq);
                        timeout = 1;
                        break;
                    }
                    __nanosleep(400);
                }
                log[sync.seq].spin_cycles = static_cast<u64>(clock64() - start);
                log[sync.seq].timeout     = timeout;
                __threadfence_system();
                sync.slot->go = sync.seq; // also releases the grid after a timeout
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const u64 start = clock64();
            while (sync.slot->go != sync.seq) {
                if (static_cast<u64>(clock64() - start) > spin_budget) {
                    tp_store(&sync.control->error[sync.side], sync.seq + 1000000ULL);
                    if (log[sync.seq].timeout == 0) { log[sync.seq].timeout = 2; }
                    break;
                }
            }
        }
        __syncthreads();
        return;
    }
    // M4c: every block spins on the peer release directly (equality gate), then the last
    // block through a second grid-wide arrival re-arms the peer's gated word to zero.
    __syncthreads();
    if (threadIdx.x == 0) {
        const u64 start = clock64();
        u64 timeout     = 0;
        while (tp_load(&sync.control->release[1 - sync.side][sync.parity]) != sync.seq) {
            if (static_cast<u64>(clock64() - start) > spin_budget) {
                tp_store(&sync.control->error[sync.side], sync.seq);
                timeout = 1;
                break;
            }
            __nanosleep(400);
        }
        if (blockIdx.x == 0) {
            log[sync.seq].spin_cycles = static_cast<u64>(clock64() - start);
            log[sync.seq].timeout     = timeout;
        }
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const u32 consumed = atomicAdd(&sync.slot->rearm, 1u);
        if (consumed + 1u == gridDim.x) {
            tp_store(&sync.control->release[1 - sync.side][sync.parity], 0);
            sync.slot->rearm = 0u;
        }
    }
    __syncthreads();
}

// Exchange kernel: identical structure to tp_link_allreduce_add_kernel plus the seq marker in
// payload[0..1] and the per-exchange instrumentation record.
__global__ void exch_add_kernel(bf16* local, size_t n,
                                volatile bf16* my_payload, volatile const bf16* peer_payload,
                                TpLinkSync sync, u64 spin_budget, ExchLog* log, int spin_mode) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        tp_store(my_payload + i, local[i]);
    }
    __threadfence_system();
    tp_arrive_and_wait(sync, spin_budget, log, spin_mode);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float sum = __bfloat162float(local[i]) +
                          __bfloat162float(tp_bf16(tp_load(peer_payload + i)));
        local[i] = __float2bfloat16(sum);
    }
    if (spin_mode == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        tp_store(&sync.control->release[1 - sync.side][sync.parity], 0); // relay-mode re-arm
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned short lo = tp_load(peer_payload + 0);
        const unsigned short hi = tp_load(peer_payload + 1);
        const float lo_f = __bfloat162float(tp_bf16(lo));
        const float hi_f = __bfloat162float(tp_bf16(hi));
        log[sync.seq].seq = sync.seq;
        log[sync.seq].consumed_payload_seq =
            static_cast<u64>(lo_f) + static_cast<u64>(hi_f) * 256ULL;
        log[sync.seq].peer_release_seen = tp_load(&sync.control->release[1 - sync.side][sync.parity]);
        log[sync.seq].own_release       = tp_load(&sync.control->release[sync.side][sync.parity]);
    }
}

// Argmax exchange: single block, mirrors tp_link_allreduce_argmax_kernel structure.
__global__ void exch_argmax_kernel(const bf16* logits, size_t n, int index_base, int* out_token,
                                   volatile bf16* my_payload, volatile const bf16* peer_payload,
                                   TpLinkSync sync, u64 spin_budget, ExchLog* log, int spin_mode) {
    float value = -1e30F;
    int index   = 0x7fffffff;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(logits[i]);
        if (v > value) { value = v; index = index_base + static_cast<int>(i); }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, value, off);
        const int   oi = __shfl_down_sync(0xffffffffu, index, off);
        if (ov > value || (ov == value && oi < index)) { value = ov; index = oi; }
    }
    __shared__ float wv[32];
    __shared__ int   wi[32];
    if ((threadIdx.x & 31u) == 0u) { wv[threadIdx.x >> 5] = value; wi[threadIdx.x >> 5] = index; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float fv = -1e30F; int fi = 0x7fffffff;
        for (int w = 0; w < static_cast<int>(blockDim.x) >> 5; ++w) {
            if (wv[w] > fv || (wv[w] == fv && wi[w] < fi)) { fv = wv[w]; fi = wi[w]; }
        }
        my_payload[0] = __float2bfloat16(fv);
        my_payload[1] = __float2bfloat16(static_cast<float>(fi & 0xffff));
        my_payload[2] = __float2bfloat16(static_cast<float>(fi >> 16));
        my_payload[3] = __float2bfloat16(static_cast<float>(sync.seq & 0xff));
        my_payload[4] = __float2bfloat16(static_cast<float>(sync.seq >> 8));
    }
    __threadfence_system();
    tp_arrive_and_wait(sync, spin_budget, log, spin_mode);
    if (threadIdx.x == 0) {
        const float pv = __bfloat162float(tp_bf16(tp_load(peer_payload + 0)));
        const int   pi = static_cast<int>(__bfloat162float(tp_bf16(tp_load(peer_payload + 1)))) +
                         (static_cast<int>(__bfloat162float(tp_bf16(tp_load(peer_payload + 2)))) << 16);
        const u64 pseq = static_cast<u64>(__bfloat162float(tp_bf16(tp_load(peer_payload + 3)))) +
                         static_cast<u64>(__bfloat162float(tp_bf16(tp_load(peer_payload + 4)))) * 256ULL;
        log[sync.seq].seq                 = sync.seq;
        log[sync.seq].consumed_payload_seq = pseq;
        log[sync.seq].peer_release_seen   = tp_load(&sync.control->release[1 - sync.side][sync.parity]);
        log[sync.seq].own_release         = tp_load(&sync.control->release[sync.side][sync.parity]);
        *out_token = pv > value || (pv == value && pi < index) ? pi : index;
        if (spin_mode == 0) {
            tp_store(&sync.control->release[1 - sync.side][sync.parity], 0); // relay-mode re-arm
        }
    }
}

// Stream-wait transport (engine TpLinkMode::StreamWait): publish kernel, driver stream
// wait on the peer release, consume kernel. No device spin; validation identical.
__global__ void publish_add_kernel(bf16* local, size_t n, volatile bf16* my_payload, TpLinkSync s) {
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
                                   TpLinkSync s, ExchLog* log) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        local[i] = __float2bfloat16(__bfloat162float(local[i]) +
                                    __bfloat162float(tp_bf16(tp_load(peer_payload + i))));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned short lo = tp_load(peer_payload + 0);
        const unsigned short hi = tp_load(peer_payload + 1);
        log[s.seq].seq = s.seq;
        log[s.seq].consumed_payload_seq =
            static_cast<u64>(__bfloat162float(tp_bf16(lo))) +
            static_cast<u64>(__bfloat162float(tp_bf16(hi))) * 256ULL;
        log[s.seq].peer_release_seen = tp_load(&s.control->release[1 - s.side][s.parity]);
        log[s.seq].own_release       = tp_load(&s.control->release[s.side][s.parity]);
        log[s.seq].spin_cycles       = 0;
        log[s.seq].timeout           = 0;
    }
}

__global__ void publish_argmax_kernel(const bf16* logits, size_t n, int index_base,
                                      volatile bf16* my_payload, TpLinkSync s) {
    float value = -1e30F;
    int index   = 0x7fffffff;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(logits[i]);
        if (v > value) { value = v; index = index_base + static_cast<int>(i); }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, value, off);
        const int   oi = __shfl_down_sync(0xffffffffu, index, off);
        if (ov > value || (ov == value && oi < index)) { value = ov; index = oi; }
    }
    __shared__ float wv[32];
    __shared__ int   wi[32];
    if ((threadIdx.x & 31u) == 0u) { wv[threadIdx.x >> 5] = value; wi[threadIdx.x >> 5] = index; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float fv = -1e30F; int fi = 0x7fffffff;
        for (int w = 0; w < static_cast<int>(blockDim.x) >> 5; ++w) {
            if (wv[w] > fv || (wv[w] == fv && wi[w] < fi)) { fv = wv[w]; fi = wi[w]; }
        }
        my_payload[0] = __float2bfloat16(fv);
        my_payload[1] = __float2bfloat16(static_cast<float>(fi & 0xffff));
        my_payload[2] = __float2bfloat16(static_cast<float>(fi >> 16));
        my_payload[3] = __float2bfloat16(static_cast<float>(s.seq & 0xff));
        my_payload[4] = __float2bfloat16(static_cast<float>(s.seq >> 8));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { // single block: arrival is trivially complete
        tp_store(&s.control->release[s.side][s.parity], s.seq);
    }
}

__global__ void consume_argmax_kernel(volatile const bf16* my_payload,
                                      volatile const bf16* peer_payload, int* out_token,
                                      TpLinkSync s, ExchLog* log) {
    if (threadIdx.x == 0) {
        const float mv = __bfloat162float(tp_bf16(tp_load(my_payload + 0)));
        const int   mi = static_cast<int>(__bfloat162float(tp_bf16(tp_load(my_payload + 1)))) +
                         (static_cast<int>(__bfloat162float(tp_bf16(tp_load(my_payload + 2)))) << 16);
        const float pv = __bfloat162float(tp_bf16(tp_load(peer_payload + 0)));
        const int   pi = static_cast<int>(__bfloat162float(tp_bf16(tp_load(peer_payload + 1)))) +
                         (static_cast<int>(__bfloat162float(tp_bf16(tp_load(peer_payload + 2)))) << 16);
        const u64 pseq = static_cast<u64>(__bfloat162float(tp_bf16(tp_load(peer_payload + 3)))) +
                         static_cast<u64>(__bfloat162float(tp_bf16(tp_load(peer_payload + 4)))) * 256ULL;
        log[s.seq].seq                 = s.seq;
        log[s.seq].consumed_payload_seq = pseq;
        log[s.seq].peer_release_seen   = tp_load(&s.control->release[1 - s.side][s.parity]);
        log[s.seq].own_release         = tp_load(&s.control->release[s.side][s.parity]);
        log[s.seq].timeout             = 0;
        *out_token = pv > mv || (pv == mv && pi < mi) ? pi : mi;
    }
}

// ---------------- representative compute kernels ----------------
__global__ void k_fill_marker(bf16* p, size_t n, u64 seq) {    const float lo = static_cast<float>(seq & 0xffu);
    const float hi = static_cast<float>((seq >> 8) & 0xffffu);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float v = static_cast<float>(i & 1023) * 0.001F;
        if (i == 0) { v = lo; } else if (i == 1) { v = hi; }
        p[i] = __float2bfloat16(v);
    }
}

__global__ void k_fill_seqf(float* p, size_t n, float v) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) { p[i] = v; }
}

// Naive shared-tiled fp32-accumulate GEMM over bf16 inputs: stands in for the engine's large
// Q4/Q5-style dequant GEMMs (dense FMA load, real bandwidth), M x K @ K x N. Each 64x64
// output tile is split across blockIdx.z halves (32 rows each) so a block stays at 1024
// threads — the original 2048-thread block was an invalid launch config that failed
// silently on every prior probe run (no error check followed those launches).
__global__ void k_gemm(const bf16* a, const bf16* b, bf16* c, int M, int K, int N) {
    __shared__ float as[64][65];
    __shared__ float bs[64][65];
    const int tile_m = blockIdx.y * 64 + blockIdx.z * 32 + (threadIdx.x / 64) * 2;
    const int tile_n = blockIdx.x * 64 + (threadIdx.x % 64);
    float acc0 = 0.F, acc1 = 0.F;
    for (int k0 = 0; k0 < K; k0 += 64) {
        const int arow0 = min(tile_m, M - 1);
        const int arow1 = min(tile_m + 1, M - 1);
        if (threadIdx.x < 64 * 16) {
            const int l = threadIdx.x / 64;
            const int cc = threadIdx.x % 64;
            as[l * 2][cc]     = __bfloat162float(a[static_cast<size_t>(arow0) * K + k0 + cc]);
            as[l * 2 + 1][cc] = __bfloat162float(a[static_cast<size_t>(arow1) * K + k0 + cc]);
            bs[cc][cc]        = __bfloat162float(b[static_cast<size_t>(k0 + cc) * N + tile_n]);
            bs[cc][(cc + 32) & 63] =
                __bfloat162float(b[static_cast<size_t>(k0 + cc) * N + min(tile_n + 32, N - 1)]);
        }
        __syncthreads();
        for (int k = 0; k < 64; ++k) {
            acc0 += as[(threadIdx.x / 64) * 2][k] * bs[k][threadIdx.x % 64];
            acc1 += as[(threadIdx.x / 64) * 2 + 1][k] * bs[k][threadIdx.x % 64];
        }
        __syncthreads();
    }
    if (tile_m < M && tile_n < N) { c[static_cast<size_t>(tile_m) * N + tile_n] = __float2bfloat16(acc0); }
    if (tile_m + 1 < M && tile_n < N) { c[static_cast<size_t>(tile_m + 1) * N + tile_n] = __float2bfloat16(acc1); }
}

// Long sequential FP32 recurrence, GDN-flavoured: few resident blocks, each walking a long
// dependent chain over the recurrence state — low occupancy, crushes WDDM DVFS like the real
// GDN decode/replay kernels.
__global__ void k_gdn_recurrence(float* state, int steps, size_t dim) {
    const size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (base >= dim) { return; }
    float s = state[base];
    for (int t = 0; t < steps; ++t) {
        s = s * 0.9999F + 0.0001F * __sinf(s * 1.0001F + static_cast<float>(t));
        state[base] = s;
    }
}

// Full-attention stand-in: each query block scans the whole KV length with fp32 dots.
__global__ void k_attn_scan(const bf16* q, const bf16* k, float* out, int kv_len, int dim) {
    const int head = blockIdx.x;
    float acc = 0.F;
    float mx  = -1e30F;
    const float qv = (head < dim) ? __bfloat162float(q[head]) : 0.F;
    for (int t = 0; t < kv_len; ++t) {
        const float kk = __bfloat162float(k[static_cast<size_t>(t) * dim + head]);
        const float score = qv * kk;
        acc += score;
        mx = fmaxf(mx, score);
    }
    out[head] = acc + mx;
}

// ---------------- host state ----------------
struct Side {
    int dev;
    cudaStream_t stream;
    cudaEvent_t flush_event;
    bf16* local;          // exchange buffer (max tokens*hidden)
    bf16* gemm_a;         // [tokens x HIDDEN]
    bf16* gemm_b;         // [HIDDEN x INTER_SHARD] weights stand-in
    bf16* gemm_c;         // [tokens x INTER_SHARD]
    bf16* gemm_b2;        // [INTER_SHARD x HIDDEN]
    bf16* gemm_c2;        // [tokens x HIDDEN]
    bf16* attn_q;
    bf16* attn_k;         // [kv_len x HIDDEN]
    float* attn_o;
    float* gdn_state;     // [HIDDEN*4]
    int* out_token;
    ExchLog* log;         // device per-exchange records
    ExchLog* hlog;        // host copy
    TpLinkDeviceSlot* slot;
    u64 seq;              // this side's exchange sequence
};

static char*    g_map_host   = nullptr;  // mapped pinned transport
static char*    g_map_dev[2] = {};       // per-device device pointers
static TpLinkControl* g_ctrl = nullptr;
static size_t   g_payload_stride = 0;
static u64      g_side_seq[2] = {0, 0};
static Side     g_side[2];

static volatile bf16* payload_ptr(int side, int parity) {
    const size_t slice = static_cast<size_t>(side * 2 + parity);
    return reinterpret_cast<volatile bf16*>(g_map_dev[side] + slice * g_payload_stride);
}

static TpLinkSync sync_for(int side, u64 seq) {
    return TpLinkSync{reinterpret_cast<TpLinkControl*>(g_map_dev[side] + g_payload_stride * 4),
                      side, static_cast<int>(seq & 1u), seq, g_side[side].slot};
}

static unsigned int exch_blocks(size_t n) {
    size_t blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks < 1) { blocks = 1; }
    if (blocks > static_cast<size_t>(MAX_BLOCKS)) { blocks = MAX_BLOCKS; }
    return static_cast<unsigned int>(blocks);
}

static void exchange_add(int side) {
    Side& s = g_side[side];
    const u64 seq   = ++g_side_seq[side];
    const int par   = static_cast<int>(seq & 1u);
    const size_t n  = static_cast<size_t>(g_tokens) * HIDDEN;
    if (!g_capturing) { CK(cudaSetDevice(s.dev)); }
    k_fill_marker<<<exch_blocks(n), BLOCK, 0, s.stream>>>(s.local, n, seq);
    if (g_gate == 0) {
        exch_add_kernel<<<exch_blocks(n), BLOCK, 0, s.stream>>>(
            s.local, n, payload_ptr(side, par), payload_ptr(1 - side, par), sync_for(side, seq),
            g_spin_budget, s.log, g_spin_mode);
        return;
    }
    const TpLinkSync sy = sync_for(side, seq);
    publish_add_kernel<<<exch_blocks(n), BLOCK, 0, s.stream>>>(s.local, n, payload_ptr(side, par), sy);
    TpLinkControl* ctrl_dev = reinterpret_cast<TpLinkControl*>(g_map_dev[side] + g_payload_stride * 4);
    CUresult r = g_wait64((CUstream)s.stream,
                          reinterpret_cast<CUdeviceptr>(
                              const_cast<u64*>(&ctrl_dev->release[1 - side][par])),
                          seq, kGeq);
    if (r != CUDA_SUCCESS) { std::fprintf(stderr, "wait64 failed: %d\n", (int)r); std::exit(1); }
    consume_add_kernel<<<exch_blocks(n), BLOCK, 0, s.stream>>>(
        s.local, n, payload_ptr(1 - side, par), sy, s.log);
}

static void exchange_argmax(int side) {
    Side& s = g_side[side];
    const u64 seq = ++g_side_seq[side];
    const int par = static_cast<int>(seq & 1u);
    if (!g_capturing) { CK(cudaSetDevice(s.dev)); }
    if (g_gate == 0) {
        exch_argmax_kernel<<<1, BLOCK, 0, s.stream>>>(
            s.local + static_cast<size_t>(g_tokens - 1) * HIDDEN, VOCAB_ROWS, side * VOCAB_ROWS,
            s.out_token, payload_ptr(side, par), payload_ptr(1 - side, par), sync_for(side, seq),
            g_spin_budget, s.log, g_spin_mode);
        return;
    }
    const TpLinkSync sy = sync_for(side, seq);
    publish_argmax_kernel<<<1, BLOCK, 0, s.stream>>>(
        s.local + static_cast<size_t>(g_tokens - 1) * HIDDEN, VOCAB_ROWS, side * VOCAB_ROWS,
        payload_ptr(side, par), sy);
    TpLinkControl* ctrl_dev = reinterpret_cast<TpLinkControl*>(g_map_dev[side] + g_payload_stride * 4);
    CUresult r = g_wait64((CUstream)s.stream,
                          reinterpret_cast<CUdeviceptr>(
                              const_cast<u64*>(&ctrl_dev->release[1 - side][par])),
                          seq, kGeq);
    if (r != CUDA_SUCCESS) { std::fprintf(stderr, "wait64 failed: %d\n", (int)r); std::exit(1); }
    consume_argmax_kernel<<<1, BLOCK, 0, s.stream>>>(
        payload_ptr(side, par), payload_ptr(1 - side, par), s.out_token, sy, s.log);
}

// One side's mirrored round: 1 embedding exchange + 2 per layer + 1 argmax, with the layer's
// compute between the exchanges. Returns the number of exchanges enqueued.
static int enqueue_round(int side) {
    Side& s = g_side[side];
    if (!g_capturing) { CK(cudaSetDevice(s.dev)); }
    const int M     = g_mode == 0 ? g_tokens : (g_tokens > 256 ? 256 : g_tokens); // prefill GEMM M cap bounds runtime
    const int kv    = 2048;
    const dim3 g1(std::min((INTER_SHARD + 63) / 64, g_gemm_blocks > 0 ? g_gemm_blocks : 136), (M + 63) / 64, 2);
    const dim3 g2(std::min((HIDDEN + 63) / 64, g_gemm_blocks > 0 ? g_gemm_blocks : 80), (M + 63) / 64, 2);
    const size_t window = static_cast<size_t>(g_tokens) * HIDDEN * sizeof(bf16);

    exchange_add(side); // embedding exchange
    for (int layer = 0; layer < LAYERS; ++layer) {
        const bool full = (layer + 1) % ATT_INTERVAL == 0;
        if (full) {
            if (g_attn) {
                k_attn_scan<<<68 * 24, BLOCK, 0, s.stream>>>(s.attn_q, s.attn_k, s.attn_o, kv, HIDDEN);
            }
            if (g_gemm) { k_gemm<<<g1, 1024, 0, s.stream>>>(s.gemm_a, s.gemm_b, s.gemm_c, M, g_gemm_k, INTER_SHARD); }
        } else {
            if (g_gemm) { k_gemm<<<g1, 1024, 0, s.stream>>>(s.gemm_a, s.gemm_b, s.gemm_c, M, g_gemm_k, INTER_SHARD); }
            if (g_gdn) {
                k_gdn_recurrence<<<48, 256, 0, s.stream>>>(s.gdn_state, g_mode == 0 ? 20000 : 60000,
                                                           static_cast<size_t>(HIDDEN) * 4);
            }
        }
        if (side == 1) { // tp_row_parallel_target on rank 1
            CK(cudaMemsetAsync(s.gemm_c2, 0, window, s.stream));
        }
        exchange_add(side); // mixer output allreduce
        if (side == 1) { // tp_row_parallel_commit on rank 1
            CK(cudaMemcpyAsync(s.gemm_a, s.gemm_c2, window, cudaMemcpyDeviceToDevice, s.stream));
        }
        if (g_gemm) {
            k_gemm<<<g1, 1024, 0, s.stream>>>(s.gemm_a, s.gemm_b, s.gemm_c, M, g_gemm_k, INTER_SHARD);
            k_gemm<<<g2, 1024, 0, s.stream>>>(s.gemm_c, s.gemm_b2, s.gemm_c2, M, INTER_SHARD, g_gemm_k);
        }
        if (side == 1) { // tp_row_parallel_target on rank 1
            CK(cudaMemsetAsync(s.gemm_c2, 0, window, s.stream));
        }
        exchange_add(side); // MLP down allreduce
        if (side == 1) { // tp_row_parallel_commit on rank 1
            CK(cudaMemcpyAsync(s.gemm_a, s.gemm_c2, window, cudaMemcpyDeviceToDevice, s.stream));
        }
    }
    if (g_gemm) { k_gemm<<<g2, 1024, 0, s.stream>>>(s.gemm_a, s.gemm_b2, s.gemm_c2, M, HIDDEN, g_gemm_k); }
    exchange_argmax(side);
    return 0;
}

static void flush_side(int side) {
    CK(cudaSetDevice(g_side[side].dev));
    CK(cudaEventRecord(g_side[side].flush_event, g_side[side].stream));
    (void)cudaEventQuery(g_side[side].flush_event);
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) { g_rounds = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            g_mode = std::atoi(argv[++i]); if (g_mode) { g_tokens = 2048; }
        }
        else if (!std::strcmp(argv[i], "--spin") && i + 1 < argc) { g_spin_budget = std::strtoull(argv[++i], nullptr, 10); }
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) { g_tokens = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--no-gemm")) { g_gemm = false; }
        else if (!std::strcmp(argv[i], "--no-gdn")) { g_gdn = false; }
        else if (!std::strcmp(argv[i], "--no-attn")) { g_attn = false; }
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) { g_threads = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--gate") && i + 1 < argc) { g_gate = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--spin-mode") && i + 1 < argc) { g_spin_mode = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--graph")) { g_graph = 1; }
        else if (!std::strcmp(argv[i], "--gemm-k") && i + 1 < argc) { g_gemm_k = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--gemm-blocks") && i + 1 < argc) { g_gemm_blocks = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--log") && i + 1 < argc) { g_max_exch_log = std::atoi(argv[++i]); }
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    int count = 0;
    CK(cudaGetDeviceCount(&count));
    if (count < 2) { std::fprintf(stderr, "need two devices\n"); return 2; }
    if (g_gate == 1) {
        void* entry = nullptr;
        if (cudaGetDriverEntryPoint("cuStreamWaitValue64", &entry, cudaEnableDefault, nullptr)
                != cudaSuccess ||
            entry == nullptr) {
            std::fprintf(stderr, "cuStreamWaitValue64 unavailable\n");
            return 2;
        }
        g_wait64 = reinterpret_cast<wait64_fn>(entry);
    }
    for (int side = 0; side < 2; ++side) {
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, g_dev[side]));
        g_name[side] = prop.name;
        std::printf("side %d = device %d (%s, SM %d.%d, %d SMs)\n", side, g_dev[side], prop.name,
                    prop.major, prop.minor, prop.multiProcessorCount);
    }

    const size_t max_elems = static_cast<size_t>(2048) * HIDDEN; // engine: prefill_chunk*hidden
    g_payload_stride        = ((max_elems * sizeof(bf16) + 63) / 64) * 64;
    const size_t total      = g_payload_stride * 4 + sizeof(TpLinkControl);
    CK(cudaHostAlloc(reinterpret_cast<void**>(&g_map_host), total,
                     cudaHostAllocMapped | cudaHostAllocPortable));
    g_ctrl = reinterpret_cast<TpLinkControl*>(g_map_host + g_payload_stride * 4);
    std::memset(g_map_host, 0, total);

    for (int side = 0; side < 2; ++side) {
        Side& s = g_side[side];
        s.dev = g_dev[side];
        CK(cudaSetDevice(s.dev));
        CK(cudaStreamCreate(&s.stream));
        CK(cudaEventCreateWithFlags(&s.flush_event, cudaEventDisableTiming));
        CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&g_map_dev[side]), g_map_host, 0));
        const size_t tokens_max = 2048;
        CK(cudaMalloc(reinterpret_cast<void**>(&s.local), tokens_max * HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gemm_a), tokens_max * HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gemm_b), static_cast<size_t>(HIDDEN) * INTER_SHARD * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gemm_c), static_cast<size_t>(tokens_max) * INTER_SHARD * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gemm_b2), static_cast<size_t>(INTER_SHARD) * HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gemm_c2), tokens_max * HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.attn_q), HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.attn_k), static_cast<size_t>(2048) * HIDDEN * sizeof(bf16)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.attn_o), HIDDEN * sizeof(float)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.gdn_state), static_cast<size_t>(HIDDEN) * 4 * sizeof(float)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.out_token), sizeof(int)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.log), static_cast<size_t>(g_max_exch_log) * sizeof(ExchLog)));
        CK(cudaMemset(s.log, 0, static_cast<size_t>(g_max_exch_log) * sizeof(ExchLog)));
        s.hlog = static_cast<ExchLog*>(std::calloc(g_max_exch_log, sizeof(ExchLog)));
        CK(cudaMalloc(reinterpret_cast<void**>(&s.slot), sizeof(TpLinkDeviceSlot) * 2));
        CK(cudaMemset(s.slot, 0, sizeof(TpLinkDeviceSlot) * 2));
        // representative fill
        k_fill_seqf<<<64, BLOCK, 0, s.stream>>>(s.gdn_state, static_cast<size_t>(HIDDEN) * 4, 1.0F);
        k_fill_marker<<<64, BLOCK, 0, s.stream>>>(s.gemm_b, static_cast<size_t>(HIDDEN) * INTER_SHARD, 7);
        k_fill_marker<<<64, BLOCK, 0, s.stream>>>(s.gemm_b2, static_cast<size_t>(INTER_SHARD) * HIDDEN, 9);
        k_fill_marker<<<64, BLOCK, 0, s.stream>>>(s.attn_k, static_cast<size_t>(2048) * HIDDEN, 3);
        CK(cudaStreamSynchronize(s.stream));
    }

    // Graph mode: capture each side's whole mirrored round into one per-rank CUDA graph
    // (the engine's decode execution shape) and replay it from two host threads. Capture
    // itself does not execute, so the two sides capture concurrently without rendezvous.
    cudaGraphExec_t graph_exec[2] = {};
    if (g_graph) {
        if (g_gate != 0) { std::fprintf(stderr, "--graph requires the spin gate (--gate 0)\n"); return 2; }
        std::memset(g_map_host, 0, total);
        g_side_seq[0] = 0; g_side_seq[1] = 0;
        cudaGraph_t graphs[2] = {};
        // Capture does not execute the kernels, so the two sides capture sequentially on
        // this thread without any rendezvous; concurrent ThreadLocal captures from two
        // threads cross-invalidate on this WDDM stack.
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(g_side[side].dev));
            CK(cudaStreamBeginCapture(g_side[side].stream, cudaStreamCaptureModeThreadLocal));
            g_capturing = true;
            enqueue_round(side);
            g_capturing = false;
            CK(cudaStreamEndCapture(g_side[side].stream, &graphs[side]));
        }
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(g_side[side].dev));
            CK(cudaGraphInstantiate(&graph_exec[side], graphs[side], 0));
            CK(cudaGraphDestroy(graphs[side]));
        }
        std::printf("captured graph rounds: %llu exchanges/side\n", g_side_seq[0]);
    }

    int first_bad_round = -1;
    for (int round = 0; round < g_rounds && first_bad_round < 0; ++round) {
        const double t0 = now_s();
        if (g_graph) {
            // Replay relies on the exchange kernels' self re-arm: no host reset between
            // rounds, exactly like the engine's captured decode schedule.
            std::thread peer([&] {
                CK(cudaSetDevice(g_side[1].dev));
                CK(cudaGraphLaunch(graph_exec[1], g_side[1].stream));
                CK(cudaStreamSynchronize(g_side[1].stream));
            });
            CK(cudaSetDevice(g_side[0].dev));
            CK(cudaGraphLaunch(graph_exec[0], g_side[0].stream));
            CK(cudaStreamSynchronize(g_side[0].stream));
            peer.join();
        } else {
        g_side_seq[0] = 0; g_side_seq[1] = 0;
        std::memset(g_map_host, 0, total); // reset control while idle
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(g_side[side].dev));
            CK(cudaMemsetAsync(g_side[side].slot, 0, sizeof(TpLinkDeviceSlot) * 2, g_side[side].stream));
            CK(cudaMemsetAsync(g_side[side].log, 0, static_cast<size_t>(g_max_exch_log) * sizeof(ExchLog), g_side[side].stream));
        }

        if (g_threads == 1 && g_gate == 0) {
            // engine lead_sync_ pattern: whole round per side, alternating devices, one thread
            enqueue_round(0);
            flush_side(0);
            enqueue_round(1);
            flush_side(1);
            bool pa = true, pb = true;
            while (pa || pb) {
                if (pa) {
                    CK(cudaSetDevice(g_side[0].dev));
                    const cudaError_t st = cudaStreamQuery(g_side[0].stream);
                    if (st != cudaErrorNotReady) { CK(st); pa = false; }
                }
                if (pb) {
                    CK(cudaSetDevice(g_side[1].dev));
                    const cudaError_t st = cudaStreamQuery(g_side[1].stream);
                    if (st != cudaErrorNotReady) { CK(st); pb = false; }
                }
            }
        } else if (g_gate == 1) {
            // engine TpProgram concurrent-enqueue pattern: both ranks' schedules enqueue at
            // once (lead + peer worker thread), each syncing its own stream at the end.
            std::thread peer([&] { enqueue_round(1); flush_side(1);
                CK(cudaSetDevice(g_side[1].dev)); CK(cudaStreamSynchronize(g_side[1].stream)); });
            enqueue_round(0);
            flush_side(0);
            CK(cudaSetDevice(g_side[0].dev));
            CK(cudaStreamSynchronize(g_side[0].stream));
            peer.join();
        } else {
            // two-thread variant for minimization comparison
            std::atomic<bool> a_done{false}, b_done{false};
            std::thread ta([&] { enqueue_round(0); flush_side(0);
                CK(cudaSetDevice(g_side[0].dev)); CK(cudaStreamSynchronize(g_side[0].stream)); a_done = true; });
            std::thread tb([&] { enqueue_round(1); flush_side(1);
                CK(cudaSetDevice(g_side[1].dev)); CK(cudaStreamSynchronize(g_side[1].stream)); b_done = true; });
            ta.join(); tb.join(); (void)a_done; (void)b_done;
        }
        }
        const double t1 = now_s();

        // pull the per-exchange logs
        for (int side = 0; side < 2; ++side) {
            CK(cudaSetDevice(g_side[side].dev));
            CK(cudaMemcpy(g_side[side].hlog, g_side[side].log,
                          static_cast<size_t>(g_max_exch_log) * sizeof(ExchLog),
                          cudaMemcpyDeviceToHost));
        }

        const u64 e0 = g_ctrl->error[0], e1 = g_ctrl->error[1];
        std::printf("round %d: %.1f ms, side_seq=%llu/%llu, error=%llu/%llu, "
                    "release a0/a1/b0/b1=%llu/%llu/%llu/%llu\n",
                    round, (t1 - t0) * 1e3, g_side_seq[0], g_side_seq[1], e0, e1,
                    g_ctrl->release[0][0], g_ctrl->release[0][1],
                    g_ctrl->release[1][0], g_ctrl->release[1][1]);
        std::fflush(stdout);

        bool bad = (e0 != 0) || (e1 != 0);
        // stale-payload check even without a timeout: consumed seq must equal exchange seq
        for (int side = 0; side < 2 && !bad; ++side) {
            for (u64 s = 1; s <= g_side_seq[side]; ++s) {
                const ExchLog& l = g_side[side].hlog[s];
                if (l.seq != s) { continue; }
                if (l.consumed_payload_seq != s || l.timeout != 0) {
                    std::printf("  ANOMALY side %d seq %llu: consumed_payload_seq=%llu "
                                "(expected %llu) timeout=%llu spin_cycles=%llu peer_rel=%llu "
                                "own_rel=%llu\n",
                                side, s, l.consumed_payload_seq, s, l.timeout, l.spin_cycles,
                                l.peer_release_seen, l.own_release);
                    bad = true;
                    break;
                }
            }
        }
        if (bad) {
            first_bad_round = round;
            // detailed dump around the first anomaly on the erroring side
            const int eside = e0 != 0 ? 0 : (e1 != 0 ? 1 : -1);
            if (eside >= 0) {
                const u64 eseq = (eside == 0 ? e0 : e1) % 1000000ULL;
                std::printf("first stall: side %d (%s), seq %llu, parity %lld, round exchange "
                            "index %lld\n", eside, g_name[eside], eseq,
                            static_cast<long long>(eseq & 1u), static_cast<long long>(eseq));
                for (u64 s = eseq > 4 ? eseq - 4 : 1; s <= eseq + 4 && s <= g_side_seq[eside]; ++s) {
                    const ExchLog& la = g_side[0].hlog[s];
                    const ExchLog& lb = g_side[1].hlog[s];
                    std::printf("  seq %3llu | side0: tmo=%llu consumed=%llu spin_cyc=%llu "
                                "peer_rel=%llu own_rel=%llu | side1: tmo=%llu consumed=%llu "
                                "spin_cyc=%llu peer_rel=%llu own_rel=%llu\n",
                                s, la.timeout, la.consumed_payload_seq, la.spin_cycles,
                                la.peer_release_seen, la.own_release, lb.timeout,
                                lb.consumed_payload_seq, lb.spin_cycles, lb.peer_release_seen,
                                lb.own_release);
                }
            }
        }
    }

    std::printf(first_bad_round < 0 ? "RESULT: no stall in %d rounds\n" : "RESULT: STALL in round %d\n",
                first_bad_round < 0 ? g_rounds : first_bad_round);
    for (int side = 0; side < 2; ++side) {
        Side& s = g_side[side];
        CK(cudaSetDevice(s.dev));
        CK(cudaStreamSynchronize(s.stream));
        if (graph_exec[side] != nullptr) { cudaGraphExecDestroy(graph_exec[side]); }
        cudaFree(s.local); cudaFree(s.gemm_a); cudaFree(s.gemm_b); cudaFree(s.gemm_c);
        cudaFree(s.gemm_b2); cudaFree(s.gemm_c2); cudaFree(s.attn_q); cudaFree(s.attn_k);
        cudaFree(s.attn_o); cudaFree(s.gdn_state); cudaFree(s.out_token); cudaFree(s.log);
        cudaFree(s.slot); cudaEventDestroy(s.flush_event); cudaStreamDestroy(s.stream);
        std::free(s.hlog);
    }
    CK(cudaFreeHost(g_map_host));
    return first_bad_round < 0 ? 0 : 1;
}
