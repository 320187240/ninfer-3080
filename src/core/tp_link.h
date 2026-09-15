#pragma once
//
// tp_link.h — raw two-GPU exchange transport for tensor parallelism over one mapped
// pinned host buffer (cudaHostAlloc Mapped|Portable), the cross-visibility fact
// measured by tools/tp_probe (Q6–Q8). Each exchange is one or two kernels per rank on
// the rank's own stream: in TpLinkMode::Spin (the TP default) one kernel per rank
// publishes, bounded-spins on the peer flag, and consumes; in TpLinkMode::StreamWait a
// publish kernel stores the payload and releases a sequence-numbered flag, a driver-API
// stream wait on the peer's flag gates the stream, and a consume kernel reads the peer
// slice. See core/tp_link.cuh for the protocols.
//
// Every exchange returns its flag word to zero when the consumer finishes (the replay
// re-arm), so StreamWait exchanges — including their cuStreamWaitValue64 gates — can be
// captured whole into CUDA graphs: the baked wait code only ever matches the current
// exchange's publish. Spin's device-side go broadcast is replay-stable for the same
// reason: its gate compares for equality, so the previous execution's higher seq left
// in the go word cannot pre-arm a replayed baked seq.
//
// The class owns no streams and no rank-local tensors: callers pass each rank's stream
// and device buffers, every op enqueues per rank (switching the current device), and
// the two ranks rendezvous per exchange. Host-side methods are sequential: the
// internal sequence must be advanced from one host thread. Ops leave the current
// device at rank_b; stream_x must belong to rank_x's device.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

// Exchange transport. Spin is the in-kernel bounded-spin rendezvous and the TP default:
// with the cv/wt payload discipline, the multi-window re-arm budget, and the WDDM clock
// holder keeping the spinning SM's clocks up, it measured 28.5 vs 16.1 tok/s on the
// two-rank decode gate (RTX 4080 + RTX 3080) — the channel-scheduler gate transition
// between publish and consume costs far more than the spin itself. StreamWait gates
// every exchange with the driver API's cuStreamWaitValue64 on the peer's mapped release
// word instead — evaluated by the GPU channel scheduler, an observer immune to SM-side
// polling pathologies — and stays available through the mode param. See core/tp_link.cuh
// for both protocols.
enum class TpLinkMode {
    Spin,
    StreamWait,
};

// Device-side handshake types, defined in core/tp_link.cuh (device intrinsics keep it
// out of host-only translation units such as the test).
struct TpLinkControl;
struct TpLinkDeviceSlot;
struct TpLinkSync;
struct TpLinkArgmaxCandidate;

/** Health snapshot read from the mapped error words. */
struct TpLinkHealth {
    // Per side: 0 when healthy; otherwise the seq of the exchange whose bounded spin
    // gave up (+1000000 when the device-side go broadcast, not the peer flag, timed out).
    std::uint64_t error[2] = {0, 0};
    // Last release sequence each side published per parity (diagnostics for a stalled
    // exchange: shows whether a side stopped publishing on one parity).
    std::uint64_t release[2][2] = {{0, 0}, {0, 0}};
    // Timeout diagnostics captured at the give-up instant (zero when healthy):
    // [side][0]=timed-out seq, [1]=spin cycles, [2]=peer release re-read, [3]=own release.
    std::uint64_t diag[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

    [[nodiscard]] bool ok() const noexcept { return error[0] == 0 && error[1] == 0; }
};

class TpLink {
public:
    // Allocates the mapped transport (payload slices for both ranks and both parities,
    // plus the control block) and the per-rank device handshake slots. max_elements is
    // the largest per-rank element count any exchange may publish.
    TpLink(int rank_a, int rank_b, std::size_t max_elements,
           TpLinkMode mode = TpLinkMode::StreamWait);
    ~TpLink();

    TpLink(const TpLink&)            = delete;
    TpLink& operator=(const TpLink&) = delete;
    TpLink(TpLink&&)                 = delete;
    TpLink& operator=(TpLink&&)      = delete;

    // local_a[i] = local_b[i] = local_a[i] + local_b[i]: bf16 payload, fp32 accumulate,
    // bf16 result on both ranks. n must be positive and fit max_elements.
    void allreduce_add(cudaStream_t stream_a, cudaStream_t stream_b, __nv_bfloat16* local_a,
                       __nv_bfloat16* local_b, std::size_t n);

    // Per-side launches for the two-rank schedule, where each rank's Text schedule enqueues
    // only its own exchange kernel on its own stream (stream order is then correct on both
    // ranks without interleaving the host passes). Each side keeps its own exchange sequence:
    // both ranks must issue the same exchanges in the same order, which the mirrored family
    // schedules guarantee. A schedule divergence surfaces as a bounded-spin timeout in
    // check_health().
    void allreduce_add_side(int side, cudaStream_t stream, __nv_bfloat16* local, std::size_t n);

    // Split-destination variant: dst[i] = src[i] + peer[i]. src == dst reduces in place;
    // src != dst publishes this rank's partial from src and lands the full sum in dst, so a
    // rank whose partial lives outside the residual stream gets the result without a
    // separate copy-back node. clear_src (src != dst only) returns src to zero in the same
    // kernel, re-arming a persistent zero-scratch accumulator for its next use.
    void allreduce_add_side(int side, cudaStream_t stream, const __nv_bfloat16* src,
                           __nv_bfloat16* dst, std::size_t n, int clear_src);

    // Per-side vocab-slice argmax; index_base is added to this side's local positions. The
    // reduced winner token is written to this side's out_token only.
    void allreduce_argmax_side(int side, cudaStream_t stream, const __nv_bfloat16* logits,
                               std::size_t n, std::int32_t index_base, std::int32_t* out_token);

    // Per-side multi-row vocab-slice argmax: logits holds `rows` row-major slices of n
    // elements each `stride` elements apart, and every row's winner is reduced across the
    // ranks into out_tokens[row] (one exchange). With id_map non-null, each row's local
    // winner position is first mapped through that device array (the rank's shortlist id
    // half) so the comparison sees global token ids; index_base is added otherwise. Equal
    // values break to the lower global id on both ranks.
    void allreduce_argmax_rows_side(int side, cudaStream_t stream, const __nv_bfloat16* logits,
                                    std::size_t n, std::size_t stride, std::int32_t rows,
                                    std::int32_t index_base, const std::int32_t* id_map,
                                    std::int32_t* out_tokens);

    // out_token_x = argmax over both vocab slices, with index_base_x added to each
    // slice's local positions by this call's kernel; equal values break to the lower
    // global index. n_a and n_b may differ (each must fit max_elements).
    void allreduce_argmax(cudaStream_t stream_a, cudaStream_t stream_b,
                          const __nv_bfloat16* logits_a, std::size_t n_a,
                          std::int32_t index_base_a, const __nv_bfloat16* logits_b,
                          std::size_t n_b, std::int32_t index_base_b,
                          std::int32_t* out_token_a, std::int32_t* out_token_b);

    // dst_a = dst_b = concat(src_a, src_b), src_a's half first on both ranks; each
    // src_x must fit max_elements. Used for column-parallel gathers. NOTE (embedding):
    // vocab-parallel row lookup composes a rank-local partial row-sum over the token ids
    // each rank owns with allreduce_add, so no dedicated exchange_rows op exists yet; add
    // one only if that full-tensor allreduce proves too wide.
    void allgather(cudaStream_t stream_a, cudaStream_t stream_b, const __nv_bfloat16* src_a,
                   const __nv_bfloat16* src_b, __nv_bfloat16* dst_a, __nv_bfloat16* dst_b,
                   std::size_t n_a, std::size_t n_b);

    // Zeroes sequence, parity, and error state for reuse after an aborted or divergent
    // schedule. Completed exchanges already leave the flag words at zero (the consume-side
    // re-arm), so this is an error-recovery path, not a per-round requirement. Safe to call
    // only while both ranks are idle: it rewrites the shared handshake state racing kernels
    // would still use.
    void reset(cudaStream_t stream_a, cudaStream_t stream_b);

    [[nodiscard]] TpLinkHealth check_health() const noexcept;

    [[nodiscard]] int rank_a() const noexcept { return ranks_[0]; }
    [[nodiscard]] int rank_b() const noexcept { return ranks_[1]; }
    [[nodiscard]] std::size_t max_elements() const noexcept { return max_elements_; }
    // Exchanges each side has started through the per-side launches (diagnostics).
    [[nodiscard]] std::uint64_t side_seq(int side) const noexcept { return side_seq_[side]; }

private:
    [[nodiscard]] TpLinkSync sync_for_(int side, std::uint64_t seq) const noexcept;
    [[nodiscard]] volatile __nv_bfloat16* payload_(int side, int parity) const noexcept;
    [[nodiscard]] volatile TpLinkArgmaxCandidate* candidate_(int side, int parity) const noexcept;
    // This rank's device mapping of the peer's release word: the address the stream-wait
    // gate polls (side's own device_base_ so the channel reads through its own mapping).
    [[nodiscard]] volatile std::uint64_t* peer_release_(int side, int parity) const noexcept;
    void reset_control_() noexcept;

    int ranks_[2]              = {0, 0};
    std::size_t max_elements_  = 0;
    std::size_t payload_stride_ = 0; // bytes per (side, parity) slice, padded to 64
    TpLinkMode mode_           = TpLinkMode::StreamWait;
    unsigned char* host_       = nullptr; // mapped pinned transport buffer, host view
    TpLinkControl* control_    = nullptr; // control block at the end of host_
    unsigned char* device_base_[2] = {};  // per-rank device mappings of host_
    TpLinkDeviceSlot* slots_[2]     = {}; // per-rank device slots, [parity]
    std::uint64_t seq_         = 0; // paired-op exchanges started; parity = seq & 1
    std::uint64_t side_seq_[2] = {0, 0}; // per-side exchanges for the *_side launches
};

// Per-rank WDDM clock holder pulse for exchange-gated decode: short spinner kernels on
// holder_stream that keep the rank's utilization signal continuous so the Windows clock
// governor does not park the SM clocks. Call from a dedicated host thread; see
// core/tp_link.cu for why the spinner must terminate (a never-ending kernel wedges the
// WDDM queue). holder_stream must belong to the rank's current device.
//
// Measured constraint (M4c): widening the pulse's SM footprint defeats its purpose. A
// 1-block spinner leaves the governor's SM-active signal too low to prevent parking
// during graph-replayed decode (both ranks observed at 210 MHz), but multi-block pulses
// contend with the round's own residency-sensitive kernels — a half-SM-count pulse
// stalls MTP3 rounds outright and 4-16-block pulses cost more than they recover. The
// 1-block shape with a short sleep gap remains the measured optimum on this stack.
void tp_clock_holder_pulse(cudaStream_t holder_stream, int iterations);

} // namespace ninfer
