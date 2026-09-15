#pragma once
//
// tp_link.cuh — device side of TpLink: the mapped-pinned spin handshake measured by
// tools/tp_probe (Q7/Q8) plus the exchange kernels built on it. See core/tp_link.h for
// the host-facing contract.
//
// Protocol per exchange (one kernel per rank, both running concurrently):
//   1. every thread publishes its slice into this rank's mapped payload region with
//      write-through stores, then __threadfence_system();
//   2. thread 0 of each block arrives on the per-parity device counter; the last block
//      releases by writing the exchange sequence number into this rank's mapped flag
//      slot;
//   3. block 0 bounded-spins on the peer's flag slot (re-armed across several budget
//      windows so a legitimately late follower rank is waited out, not consumed stale),
//      then broadcasts a device-side go flag; every other block bounded-spins on that go
//      flag;
//   4. all threads consume the peer payload. Parity double buffering of payload+flag
//      slots lets back-to-back exchanges pipeline.
//
// Mapped-pinned coherence: plain volatile accesses bypass L1 but may be served from a
// stale L2 line on the reading GPU (observed as a >1 s stuck flag while the peer had long
// published), so every access that crosses the transport uses ld.global.cv (__ldcv, never
// cache) for reads and st.global.wt (__stwt, write-through) for writes. Only the strictly
// device-local go broadcast keeps plain volatile semantics.
//

#include <cuda_bf16.h>

#include <cstddef>
#include <cstring>
#include <cstdint>

namespace ninfer {

// Bounded-spin budget in clock64() cycles, per re-arm window. Sized for two WDDM realities:
// display scheduling stalls on a desktop-attached rank (>60 ms observed) and first-launch module
// uploads on the lagging rank while the peer runs ahead on arrival-based releases.
// ~1 s at 2.5 GHz; a timeout writes this rank's error word instead of hanging.
inline constexpr std::uint64_t kTpLinkSpinCycles = 2500000000ULL;

// Re-arm windows before an exchange finally gives up. The follower rank legitimately lags by
// multiple seconds during the first rounds of a run (first-launch module loads of the decode
// schedule plus WDDM DVFS cold start), and a single-window give-up consumed the peer's stale
// parity payload there — one corrupted exchange per early round, garbage output (M3b-2). The
// window budget scales with the spinning SM's clock, so this is ~60 s at boost and far more at
// the idle clocks a lone spinner drops to; the hard cap only exists so a schedule divergence
// (a rank that stopped issuing exchanges) still terminates. Recovered waits leave error clear.
inline constexpr int kTpLinkSpinWindows = 60;

// Control words inside the mapped transport buffer. Kernels access them only through
// ld.cv/st.wt; TpLink::reset() zeroes them from the host while both ranks idle.
struct TpLinkControl {
    volatile std::uint64_t release[2][2]; // [side][parity] = seq once that side published
    volatile std::uint64_t error[2];      // [side] nonzero after a bounded-spin timeout
    // Timeout diagnostics, written only on the give-up path of tp_link_arrive_and_wait:
    // [side][0]=timed-out seq, [1]=spin cycles consumed, [2]=peer release re-read at give-up,
    // [3]=own release re-read at give-up (was this rank itself still unpublished?).
    volatile std::uint64_t diag[2][4];
};

// Per-rank device handshake state. The arrival counter and go broadcast never cross the
// PCIe link, so they live in regular device memory allocated by the host class.
struct TpLinkDeviceSlot {
    volatile std::uint64_t go; // block 0 sets this to seq once the peer release was observed
    std::uint32_t arrive;      // grid-wide arrival counter; the last block resets it to zero
};

// One exchange's handshake endpoints for one rank. Payload pointers are handed to the
// kernels separately because their types differ per operation.
struct TpLinkSync {
    TpLinkControl* control; // this rank's device mapping of the shared control block
    int side;               // 0 = rank_a, 1 = rank_b
    int parity;             // seq & 1: double-buffered payload/flag selector
    std::uint64_t seq;      // exchange sequence number, from 1, strictly increasing
    TpLinkDeviceSlot* slot; // this rank's device slot for this parity
};

// ld.cv read of a cross-transport word: always fetched from system memory, never a cached
// line (the peer GPU's writes are only visible at the system point of coherence).
__device__ __forceinline__ std::uint64_t tp_link_load(const volatile std::uint64_t* address) {
    unsigned long long value = 0;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(value) : "l"(address));
    return static_cast<std::uint64_t>(value);
}

// st.wt write of a cross-transport word: write-through to system memory.
__device__ __forceinline__ void tp_link_store(volatile std::uint64_t* address,
                                              std::uint64_t value) {
    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ unsigned short tp_link_load(const volatile __nv_bfloat16* address) {
    unsigned short value = 0;
    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(value) : "l"(address));
    return value;
}

__device__ __forceinline__ void tp_link_store(volatile __nv_bfloat16* address, __nv_bfloat16 value) {
    const unsigned short bits = __bfloat16_as_ushort(value);
    asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(address), "h"(bits) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 tp_link_bfloat16(unsigned short bits) {
    return __ushort_as_bfloat16(bits);
}

// Vocab-slice winner exchanged between ranks.
struct TpLinkArgmaxCandidate {
    float value;          // fp32 promotion of the bf16 logit
    std::int32_t index;   // global vocab position (slice base already added)
};

// Cross-transport candidate access (fp32 value + i32 index in one 8-byte word): the same
// cv/wt discipline as every other payload access. A plain volatile read here can be served
// from a stale L2 line holding the previous parity-round's bytes — the argmax candidate is
// the token producer, so a stale read surfaces as garbage sampled tokens (observed
// end-to-end while small L2-cold unit tests stayed exact).
__device__ __forceinline__ TpLinkArgmaxCandidate tp_link_load(
    const volatile TpLinkArgmaxCandidate* address) {
    const std::uint64_t bits = tp_link_load(reinterpret_cast<const volatile std::uint64_t*>(address));
    TpLinkArgmaxCandidate candidate;
    std::memcpy(&candidate, &bits, sizeof candidate);
    return candidate;
}

__device__ __forceinline__ void tp_link_store(volatile TpLinkArgmaxCandidate* address,
                                              const TpLinkArgmaxCandidate& candidate) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &candidate, sizeof bits);
    tp_link_store(reinterpret_cast<volatile std::uint64_t*>(address), bits);
}

// Completes one side of an exchange after the caller published its payload: grid-wide
// arrival, release, bounded spin on the peer release, device-side go broadcast. Every
// thread of the grid must call this, each after fencing its own payload stores. On
// a timeout the error word is written and the exchange completes with unusable peer data;
// the host observes that through TpLink::check_health(). The grid must be small enough
// that every block is concurrently resident (TpLink caps it) or the arrival stalls.
//
// The peer observation is relayed through one block (block 0 spins on the peer flag, the
// rest of the grid spins on the device-local go word) instead of every block polling the
// peer flag directly: measured at engine round shape (tools/tp_probe tp_engine_repro
// --graph, 130 exchanges interleaved with compute), one ld.global.cv poller per block
// congests the PCIe link — the 4080's x4 slot — and costs ~+17 us per exchange versus the
// relay (34.7 vs 32.5 ms per round; the eager shape regresses the same way).
__device__ __forceinline__ void tp_link_arrive_and_wait(TpLinkSync sync) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int arrived = atomicAdd(&sync.slot->arrive, 1u);
        if (arrived + 1u == gridDim.x) {
            tp_link_store(&sync.control->release[sync.side][sync.parity], sync.seq); // release
            sync.slot->arrive = 0u; // no arrivals remain this kernel; next user is a later kernel
        }
    }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            // Re-armed bounded spin on the peer release: a follower rank that is legitimately
            // still in flight (first-round module loads, DVFS cold start) is waited out rather
            // than consumed stale; only the final expired window gives up.
            for (int window = 0;; ++window) {
                const std::uint64_t start = clock64();
                while (tp_link_load(&sync.control->release[1 - sync.side][sync.parity]) <
                       sync.seq) {
                    if (static_cast<std::uint64_t>(clock64() - start) > kTpLinkSpinCycles) {
                        break; // window expired
                    }
                    __nanosleep(400);
                }
                if (tp_link_load(&sync.control->release[1 - sync.side][sync.parity]) >= sync.seq) {
                    break; // observed
                }
                if (window + 1 >= kTpLinkSpinWindows) {
                    tp_link_store(&sync.control->error[sync.side], sync.seq);
                    // Diagnostics at the give-up instant: separates "peer published but this
                    // rank never observed it" (peer value >= seq here) from "peer had not
                    // published yet" (peer value < seq here, possibly showing how far behind).
                    tp_link_store(&sync.control->diag[sync.side][0], sync.seq);
                    tp_link_store(&sync.control->diag[sync.side][1],
                                  static_cast<std::uint64_t>(clock64() - start));
                    tp_link_store(&sync.control->diag[sync.side][2],
                                  tp_link_load(&sync.control->release[1 - sync.side][sync.parity]));
                    tp_link_store(&sync.control->diag[sync.side][3],
                                  tp_link_load(&sync.control->release[sync.side][sync.parity]));
                    break;
                }
            }
            __threadfence_system();
            sync.slot->go = sync.seq; // also releases the grid after a timeout
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // The go wait must tolerate every window block 0's peer spin can consume (plus jitter
        // for the nanosleep cadence), or a non-block-0 block would race past block 0 into the
        // peer payload before block 0 observed the release. The gate compares for equality,
        // not >=: a captured replay re-runs the same baked seq, and the go word still holds
        // the previous execution's (higher) seq at kernel start, so a >= wait would be
        // pre-armed and let the grid race past block 0's peer observation.
        for (int window = 0;; ++window) {
            const std::uint64_t start = clock64();
            while (sync.slot->go != sync.seq) {
                if (static_cast<std::uint64_t>(clock64() - start) > kTpLinkSpinCycles) {
                    break; // window expired
                }
            }
            if (sync.slot->go == sync.seq) { break; }
            if (window + 1 >= kTpLinkSpinWindows + 2) {
                tp_link_store(&sync.control->error[sync.side], sync.seq + 1000000ULL);
                break;
            }
        }
    }
    __syncthreads();
}

// dst[i] = src[i] + peer[i] elementwise: bf16 payload, fp32 accumulate, bf16 result.
// src == dst is the in-place allreduce; src != dst publishes one rank's bare partial and
// lands the full sum straight in the residual stream, skipping the scratch copy-back the
// peer rank's commit used to need. When clear_src is set, the consume returns src to zero
// after reading it, re-arming a persistent zero-scratch buffer for its next leaf without
// a separate memset node (the buffer is zeroed once at startup; every later user of the
// value leaves it zero again, so captured replays stay self-contained).
__global__ void tp_link_allreduce_add_kernel(__nv_bfloat16* src, __nv_bfloat16* dst,
                                             std::size_t n,
                                             volatile __nv_bfloat16* my_payload,
                                             volatile const __nv_bfloat16* peer_payload,
                                             TpLinkSync sync, int clear_src) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        tp_link_store(my_payload + i, src[i]);
    }
    __threadfence_system();
    tp_link_arrive_and_wait(sync);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float sum =
            __bfloat162float(src[i]) +
            __bfloat162float(tp_link_bfloat16(tp_link_load(peer_payload + i)));
        dst[i] = __float2bfloat16(sum);
        if (clear_src != 0 && src != dst) { src[i] = __float2bfloat16(0.0F); }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tp_link_store(&sync.control->release[1 - sync.side][sync.parity],
                      0); // re-arm the gated word for the slot's next use
    }
}

// Global argmax over two vocab slices. index_base selects this slice's global offset;
// equal values break to the lower global index, identically on both ranks. Single
// block: the handshake is grid-wide anyway and one block reduces the local winner
// without extra cross-block state.
__global__ void tp_link_allreduce_argmax_kernel(const __nv_bfloat16* logits, std::size_t n,
                                                std::int32_t index_base, std::int32_t* out_token,
                                                volatile TpLinkArgmaxCandidate* my_payload,
                                                volatile const TpLinkArgmaxCandidate* peer_payload,
                                                TpLinkSync sync) {
    float value          = __int_as_float(0xff800000u); // -inf: an empty slice never wins
    std::int32_t index   = 0x7fffffff;
    const int warp_count = static_cast<int>(blockDim.x) >> 5;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float candidate_value  = __bfloat162float(logits[i]);
        const std::int32_t candidate_index = index_base + static_cast<std::int32_t>(i);
        if (candidate_value > value || (candidate_value == value && candidate_index < index)) {
            value = candidate_value;
            index = candidate_index;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(0xffffffffu, value, offset);
        const std::int32_t other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (other_value > value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
    __shared__ TpLinkArgmaxCandidate warp_winners[32];
    if ((threadIdx.x & 31u) == 0u) {
        warp_winners[threadIdx.x >> 5].value = value;
        warp_winners[threadIdx.x >> 5].index = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        TpLinkArgmaxCandidate winner{__int_as_float(0xff800000u), 0x7fffffff};
        for (int warp = 0; warp < warp_count; ++warp) {
            if (warp_winners[warp].value > winner.value ||
                (warp_winners[warp].value == winner.value && warp_winners[warp].index < winner.index)) {
                winner = warp_winners[warp];
            }
        }
        tp_link_store(my_payload, winner);
    }
    __threadfence_system();
    tp_link_arrive_and_wait(sync);
    if (threadIdx.x == 0) {
        const TpLinkArgmaxCandidate mine = tp_link_load(my_payload);
        const TpLinkArgmaxCandidate peer = tp_link_load(peer_payload);
        const bool peer_wins =
            peer.value > mine.value || (peer.value == mine.value && peer.index < mine.index);
        *out_token = peer_wins ? peer.index : mine.index;
        tp_link_store(&sync.control->release[1 - sync.side][sync.parity],
                      0); // re-arm the gated word for the slot's next use
    }
}

// Multi-row vocab-slice argmax (one block per row; MTP candidate verify and shortlist
// proposals). Each block reduces its own row's local winner — optionally mapped through
// this rank's shortlist id table before the comparison — publishes it at candidate[row],
// and after the rendezvous compares against the peer's row winner: equal values break to
// the lower global id, identically on both ranks.
__global__ void tp_link_allreduce_argmax_rows_kernel(const __nv_bfloat16* logits, std::size_t n,
                                                     std::size_t stride, std::int32_t rows,
                                                     std::int32_t index_base,
                                                     const std::int32_t* id_map,
                                                     std::int32_t* out_tokens,
                                                     volatile TpLinkArgmaxCandidate* my_payload,
                                                     volatile const TpLinkArgmaxCandidate* peer_payload,
                                                     TpLinkSync sync) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    float value          = __int_as_float(0xff800000u); // -inf: an empty slice never wins
    std::int32_t index   = 0x7fffffff;
    const int warp_count = static_cast<int>(blockDim.x) >> 5;
    const __nv_bfloat16* row_logits = logits + static_cast<std::size_t>(row) * stride;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float candidate_value       = __bfloat162float(row_logits[i]);
        const std::int32_t candidate_index = id_map != nullptr
                                                 ? id_map[i]
                                                 : index_base + static_cast<std::int32_t>(i);
        if (candidate_value > value || (candidate_value == value && candidate_index < index)) {
            value = candidate_value;
            index = candidate_index;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(0xffffffffu, value, offset);
        const std::int32_t other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (other_value > value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
    __shared__ TpLinkArgmaxCandidate warp_winners[32];
    if ((threadIdx.x & 31u) == 0u) {
        warp_winners[threadIdx.x >> 5].value = value;
        warp_winners[threadIdx.x >> 5].index = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        TpLinkArgmaxCandidate winner{__int_as_float(0xff800000u), 0x7fffffff};
        for (int warp = 0; warp < warp_count; ++warp) {
            if (warp_winners[warp].value > winner.value ||
                (warp_winners[warp].value == winner.value && warp_winners[warp].index < winner.index)) {
                winner = warp_winners[warp];
            }
        }
        tp_link_store(my_payload + row, winner);
    }
    __threadfence_system();
    tp_link_arrive_and_wait(sync);
    if (threadIdx.x == 0) {
        const TpLinkArgmaxCandidate mine = tp_link_load(my_payload + row);
        const TpLinkArgmaxCandidate peer = tp_link_load(peer_payload + row);
        const bool peer_wins =
            peer.value > mine.value || (peer.value == mine.value && peer.index < mine.index);
        out_tokens[row] = peer_wins ? peer.index : mine.index;
        if (blockIdx.x == 0) {
            tp_link_store(&sync.control->release[1 - sync.side][sync.parity],
                          0); // re-arm the gated word for the slot's next use
        }
    }
}

// dst = concat(src_a, src_b) on both ranks: src_a's half is always first, so each rank
// places its local half at local_offset (0 for rank_a, n_a for rank_b). The local half
// is copied device-side directly from src; only the peer half crosses the transport.
__global__ void tp_link_allgather_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                         std::size_t n_local, std::size_t local_offset,
                                         std::size_t n_peer, std::size_t peer_offset,
                                         volatile __nv_bfloat16* my_payload,
                                         volatile const __nv_bfloat16* peer_payload,
                                         TpLinkSync sync) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_local; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        tp_link_store(my_payload + i, src[i]);
    }
    __threadfence_system();
    tp_link_arrive_and_wait(sync);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_local; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        dst[local_offset + i] = src[i];
    }
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_peer; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        dst[peer_offset + i] = tp_link_bfloat16(tp_link_load(peer_payload + i));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tp_link_store(&sync.control->release[1 - sync.side][sync.parity],
                      0); // re-arm the gated word for the slot's next use
    }
}

// ---- Stream-wait gate --------------------------------------------------------
//
// The alternative exchange mode. The spin handshake above had a WDDM failure mode that
// pushed this mode to the default seat once: a rank's ld.global.cv poll of the peer's
// mapped release word kept returning a stale value for far longer than any usable cycle
// budget (observed: >60 re-arm windows) while the host CPU read the same word as
// published. That observation decomposed into two causes with kernel/host-side fixes —
// the spinning SM dropping to idle clocks (now held up by the TpProgram clock holder)
// and the go broadcast pre-arming under captured replays (now the equality gate above)
// — after which the spin measured faster end to end than this gate (28.5 vs 16.1 tok/s
// decode): each stream-wait exchange pays a channel-scheduler gate transition between
// the publish and consume kernels, and ~130 of them per decode round dominate that cost.
// This mode splits each exchange into publish and consume kernels with NO device spin
// and gates them with the driver API's cuStreamWaitValue64 on the peer release word:
// the wait is evaluated by the GPU channel scheduler (the same system-memory observer
// class as the CPU, verified by tools/tp_probe/waitvalue_drv_probe.cu) and, unlike a
// host-callback gate, it is a plain stream operation — the host can keep enqueueing a
// whole round ahead without deadlocking on the WDDM software queue. cuStreamWaitValue64
// is also CUDA-graph capturable (tools/tp_probe/waitvalue_graph_probe.cu, Q13), so a
// captured decode round carries its gates.
//
// Ordering: each rank wt-stores its payload, fences, then stores its release; the gate
// observing both releases therefore implies both payloads are at the system coherence
// point, and the consume kernel's cv loads fetch them — the same ordering the spin path
// relies on, with the observer moved off the SM. There is no timeout: a schedule
// divergence hangs the stream instead of consuming stale data, which the mirrored family
// schedules make a programming error rather than a runtime condition.
//
// Replay re-arm: the wait value is a graph-node parameter baked at capture time, so a
// replayed exchange always gates on the same code its publish kernel writes. The consume
// kernel returns the gated word to zero after loading the payload, which makes every
// exchange self-contained: the word is 0 between exchanges (eager or captured, any
// interleaving), so a gate can only pass on the current exchange's publish even though
// the codes repeat across replays. The peer's next publish on the slot is stream-ordered
// behind a gate chain that includes this kernel's completion, so the reset can never
// overwrite a fresh release. Codes are unique per exchange instance because they come
// from the per-side sequence counter, which advances across captures and eager exchanges
// alike. Publish tail shared by every stream-wait exchange: grid-wide arrival, last block
// releases. Callers store+__threadfence_system() their payload first.
__device__ __forceinline__ void tp_link_publish(TpLinkSync sync) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int arrived = atomicAdd(&sync.slot->arrive, 1u);
        if (arrived + 1u == gridDim.x) {
            tp_link_store(&sync.control->release[sync.side][sync.parity], sync.seq); // release
            sync.slot->arrive = 0u; // no arrivals remain this kernel; next user is a later kernel
        }
    }
}

__global__ void tp_link_allreduce_add_publish(const __nv_bfloat16* src, std::size_t n,
                                              volatile __nv_bfloat16* my_payload,
                                              TpLinkSync sync) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        tp_link_store(my_payload + i, src[i]);
    }
    __threadfence_system();
    tp_link_publish(sync);
}

__global__ void tp_link_allreduce_add_consume(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                              std::size_t n,
                                              volatile const __nv_bfloat16* peer_payload,
                                              volatile std::uint64_t* peer_release) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float sum =
            __bfloat162float(src[i]) +
            __bfloat162float(tp_link_bfloat16(tp_link_load(peer_payload + i)));
        dst[i] = __float2bfloat16(sum);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tp_link_store(peer_release, 0); // re-arm the gated word for the slot's next use
    }
}

// Single block, same reduction as the spin variant; only the candidate store + release
// happen here, the cross-rank comparison moves to the consume kernel.
__global__ void tp_link_allreduce_argmax_publish(const __nv_bfloat16* logits, std::size_t n,
                                                 std::int32_t index_base,
                                                 volatile TpLinkArgmaxCandidate* my_payload,
                                                 TpLinkSync sync) {
    float value          = __int_as_float(0xff800000u); // -inf: an empty slice never wins
    std::int32_t index   = 0x7fffffff;
    const int warp_count = static_cast<int>(blockDim.x) >> 5;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float candidate_value  = __bfloat162float(logits[i]);
        const std::int32_t candidate_index = index_base + static_cast<std::int32_t>(i);
        if (candidate_value > value || (candidate_value == value && candidate_index < index)) {
            value = candidate_value;
            index = candidate_index;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(0xffffffffu, value, offset);
        const std::int32_t other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (other_value > value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
    __shared__ TpLinkArgmaxCandidate warp_winners[32];
    if ((threadIdx.x & 31u) == 0u) {
        warp_winners[threadIdx.x >> 5].value = value;
        warp_winners[threadIdx.x >> 5].index = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        TpLinkArgmaxCandidate winner{__int_as_float(0xff800000u), 0x7fffffff};
        for (int warp = 0; warp < warp_count; ++warp) {
            if (warp_winners[warp].value > winner.value ||
                (warp_winners[warp].value == winner.value && warp_winners[warp].index < winner.index)) {
                winner = warp_winners[warp];
            }
        }
        tp_link_store(my_payload, winner);
    }
    __threadfence_system();
    tp_link_publish(sync);
}

__global__ void tp_link_allreduce_argmax_consume(
    volatile const TpLinkArgmaxCandidate* my_payload,
    volatile const TpLinkArgmaxCandidate* peer_payload, std::int32_t* out_token,
    volatile std::uint64_t* peer_release) {
    if (threadIdx.x == 0) {
        const TpLinkArgmaxCandidate mine = tp_link_load(my_payload);
        const TpLinkArgmaxCandidate peer = tp_link_load(peer_payload);
        const bool peer_wins =
            peer.value > mine.value || (peer.value == mine.value && peer.index < mine.index);
        *out_token = peer_wins ? peer.index : mine.index;
        tp_link_store(peer_release, 0); // re-arm the gated word for the slot's next use
    }
}

__global__ void tp_link_allgather_publish(const __nv_bfloat16* src, std::size_t n_local,
                                          volatile __nv_bfloat16* my_payload, TpLinkSync sync) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_local; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        tp_link_store(my_payload + i, src[i]);
    }
    __threadfence_system();
    tp_link_publish(sync);
}

__global__ void tp_link_allgather_consume(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                          std::size_t n_local, std::size_t local_offset,
                                          std::size_t n_peer, std::size_t peer_offset,
                                          volatile const __nv_bfloat16* peer_payload,
                                          volatile std::uint64_t* peer_release) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_local; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        dst[local_offset + i] = src[i];
    }
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_peer; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        dst[peer_offset + i] = tp_link_bfloat16(tp_link_load(peer_payload + i));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        tp_link_store(peer_release, 0); // re-arm the gated word for the slot's next use
    }
}

} // namespace ninfer
