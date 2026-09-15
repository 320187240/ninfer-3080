#include "core/tp_link.h"

#include "core/device.h" // CUDA_CHECK
#include "core/tp_link.cuh"

#include <cuda.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

constexpr int kBlockThreads      = 256;
constexpr int kMaxExchangeBlocks = 64; // keeps every block of both ranks' kernels resident
                                       // (the smaller GPU has 68 SMs), which the spin needs

// Grid for payload-carrying exchanges: one block per 256 elements, clamped so the whole
// grid is concurrently resident; the arrival handshake would stall otherwise.
unsigned int exchange_blocks(std::size_t n) {
    std::size_t blocks = (n + kBlockThreads - 1) / kBlockThreads;
    if (blocks < 1) { blocks = 1; }
    if (blocks > static_cast<std::size_t>(kMaxExchangeBlocks)) { blocks = kMaxExchangeBlocks; }
    return static_cast<unsigned int>(blocks);
}

// ---- Stream-wait gate --------------------------------------------------------
//
// cuStreamWaitValue64 (driver API; the runtime surface was removed in CUDA 13) blocks this
// rank's stream until the PEER's mapped release word reaches this exchange's sequence
// number. The channel scheduler evaluates the wait — the observer class the SM cv spin
// needed the WDDM clock holder to match (see core/tp_link.cuh) — and as
// a plain stream operation it never blocks the host, so whole-round enqueue-ahead cannot
// deadlock on the WDDM software queue the way a host-callback gate can.
using TpLinkWaitValue64 = CUresult(CUDAAPI*)(CUstream, CUdeviceptr, cuuint64_t,
                                             unsigned int);

constexpr unsigned int kTpLinkWaitGeq = 0x0; // CU_STREAM_WAIT_VALUE_GEQ

TpLinkWaitValue64 tp_link_wait_value64() {
    static TpLinkWaitValue64 entry = [] {
        void* resolved = nullptr;
        const cudaError_t err =
            cudaGetDriverEntryPoint("cuStreamWaitValue64", &resolved, cudaEnableDefault, nullptr);
        if (err != cudaSuccess || resolved == nullptr) {
            throw std::runtime_error(
                cuda_error_message("TpLink needs the cuStreamWaitValue64 driver entry point", err));
        }
        return reinterpret_cast<TpLinkWaitValue64>(resolved);
    }();
    return entry;
}

void check_cu(CUresult err, const char* what) {
    if (err != CUDA_SUCCESS) {
        // Driver-entry strings are not linked; the numeric CUresult is enough to diagnose.
        throw std::runtime_error(std::string(what) + " failed with CUresult " +
                                 std::to_string(static_cast<int>(err)));
    }
}

} // namespace

TpLink::TpLink(int rank_a, int rank_b, std::size_t max_elements, TpLinkMode mode)
    : ranks_{rank_a, rank_b}, max_elements_(max_elements), mode_(mode) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("TpLink cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("TpLink needs at least one CUDA device"); }
    for (int side = 0; side < 2; ++side) {
        if (ranks_[side] < 0 || ranks_[side] >= count) {
            throw std::runtime_error("TpLink rank is not a valid CUDA device id");
        }
    }
    if (rank_a == rank_b) { throw std::runtime_error("TpLink ranks must be two distinct devices"); }
    if (max_elements_ == 0) { throw std::runtime_error("TpLink max_elements must be positive"); }

    // Transport layout: payload[side][parity] slices (parity double buffering is what
    // lets back-to-back exchanges pipeline), then the control block on its own 64-byte
    // line. One mapped allocation covers both; each rank maps all four slices.
    payload_stride_ = ((max_elements_ * sizeof(__nv_bfloat16) + 63) / 64) * 64;
    const std::size_t payload_bytes = payload_stride_ * 4;
    const std::size_t total_bytes   = payload_bytes + sizeof(TpLinkControl);

    int current_device = 0;
    const cudaError_t current_err = cudaGetDevice(&current_device);

    err = cudaHostAlloc(reinterpret_cast<void**>(&host_), total_bytes,
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        host_ = nullptr;
        throw std::runtime_error(
            cuda_error_message("TpLink cudaHostAlloc(Mapped|Portable) failed", err));
    }
    control_ = reinterpret_cast<TpLinkControl*>(host_ + payload_bytes);
    reset_control_();

    for (int side = 0; side < 2; ++side) {
        err = cudaSetDevice(ranks_[side]);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_base_[side]), host_, 0);
        }
        if (err == cudaSuccess) {
            err = cudaMalloc(reinterpret_cast<void**>(&slots_[side]),
                             sizeof(TpLinkDeviceSlot) * 2);
        }
        if (err == cudaSuccess) { err = cudaMemset(slots_[side], 0, sizeof(TpLinkDeviceSlot) * 2); }
        if (err != cudaSuccess) {
            if (slots_[side] != nullptr) {
                cudaSetDevice(ranks_[side]);
                log_cuda_error("cudaFree", cudaFree(slots_[side]));
                slots_[side] = nullptr;
            }
            log_cuda_error("cudaFreeHost", cudaFreeHost(host_));
            host_ = nullptr;
            control_ = nullptr;
            device_base_[0] = nullptr;
            device_base_[1] = nullptr;
            throw std::runtime_error(
                cuda_error_message("TpLink per-rank device state allocation failed", err));
        }
    }
    if (current_err == cudaSuccess) { log_cuda_error("cudaSetDevice", cudaSetDevice(current_device)); }
}

TpLink::~TpLink() {
    for (int side = 0; side < 2; ++side) {
        if (slots_[side] != nullptr) {
            log_cuda_error("cudaSetDevice", cudaSetDevice(ranks_[side]));
            log_cuda_error("cudaFree", cudaFree(slots_[side]));
            slots_[side] = nullptr;
        }
    }
    if (host_ != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(host_));
        host_    = nullptr;
        control_ = nullptr;
    }
}

void TpLink::allreduce_add(cudaStream_t stream_a, cudaStream_t stream_b, __nv_bfloat16* local_a,
                           __nv_bfloat16* local_b, std::size_t n) {
    if (n == 0 || n > max_elements_) {
        throw std::invalid_argument("TpLink allreduce_add n must be in (0, max_elements]");
    }
    const std::uint64_t seq      = ++seq_;
    const int parity             = static_cast<int>(seq & 1u);
    const unsigned int blocks    = exchange_blocks(n);
    const TpLinkSync sync_a      = sync_for_(0, seq);
    const TpLinkSync sync_b      = sync_for_(1, seq);

    if (mode_ == TpLinkMode::Spin) {
        CUDA_CHECK(cudaSetDevice(ranks_[0]));
        tp_link_allreduce_add_kernel<<<blocks, kBlockThreads, 0, stream_a>>>(
            local_a, local_a, n, payload_(0, parity), payload_(1, parity), sync_a, 0);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaSetDevice(ranks_[1]));
        tp_link_allreduce_add_kernel<<<blocks, kBlockThreads, 0, stream_b>>>(
            local_b, local_b, n, payload_(1, parity), payload_(0, parity), sync_b, 0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Both publishes go in before the gates: neither publish blocks, so both releases can
    // land while each gate polls.
    CUDA_CHECK(cudaSetDevice(ranks_[0]));
    tp_link_allreduce_add_publish<<<blocks, kBlockThreads, 0, stream_a>>>(
        local_a, n, payload_(0, parity), sync_a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(ranks_[1]));
    tp_link_allreduce_add_publish<<<blocks, kBlockThreads, 0, stream_b>>>(
        local_b, n, payload_(1, parity), sync_b);
    CUDA_CHECK(cudaGetLastError());

    for (int side = 0; side < 2; ++side) {
        CUDA_CHECK(cudaSetDevice(ranks_[side]));
        __nv_bfloat16* local            = side == 0 ? local_a : local_b;
        cudaStream_t stream             = side == 0 ? stream_a : stream_b;
        check_cu(tp_link_wait_value64()((CUstream)stream,
                                        reinterpret_cast<CUdeviceptr>(peer_release_(side, parity)),
                                        seq, kTpLinkWaitGeq),
                 "TpLink cuStreamWaitValue64");
        tp_link_allreduce_add_consume<<<blocks, kBlockThreads, 0, stream>>>(
            local, local, n, payload_(1 - side, parity), peer_release_(side, parity));
        CUDA_CHECK(cudaGetLastError());
    }
}

void TpLink::allreduce_argmax(cudaStream_t stream_a, cudaStream_t stream_b,
                              const __nv_bfloat16* logits_a, std::size_t n_a,
                              std::int32_t index_base_a, const __nv_bfloat16* logits_b,
                              std::size_t n_b, std::int32_t index_base_b,
                              std::int32_t* out_token_a, std::int32_t* out_token_b) {
    if (n_a > max_elements_ || n_b > max_elements_) {
        throw std::invalid_argument("TpLink allreduce_argmax slice exceeds max_elements");
    }
    const std::uint64_t seq = ++seq_;
    const int parity        = static_cast<int>(seq & 1u);
    const TpLinkSync sync_a = sync_for_(0, seq);
    const TpLinkSync sync_b = sync_for_(1, seq);

    if (mode_ == TpLinkMode::Spin) {
        CUDA_CHECK(cudaSetDevice(ranks_[0]));
        tp_link_allreduce_argmax_kernel<<<1, kBlockThreads, 0, stream_a>>>(
            logits_a, n_a, index_base_a, out_token_a, candidate_(0, parity), candidate_(1, parity),
            sync_a);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaSetDevice(ranks_[1]));
        tp_link_allreduce_argmax_kernel<<<1, kBlockThreads, 0, stream_b>>>(
            logits_b, n_b, index_base_b, out_token_b, candidate_(1, parity), candidate_(0, parity),
            sync_b);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(cudaSetDevice(ranks_[0]));
    tp_link_allreduce_argmax_publish<<<1, kBlockThreads, 0, stream_a>>>(
        logits_a, n_a, index_base_a, candidate_(0, parity), sync_a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(ranks_[1]));
    tp_link_allreduce_argmax_publish<<<1, kBlockThreads, 0, stream_b>>>(
        logits_b, n_b, index_base_b, candidate_(1, parity), sync_b);
    CUDA_CHECK(cudaGetLastError());

    for (int side = 0; side < 2; ++side) {
        CUDA_CHECK(cudaSetDevice(ranks_[side]));
        cudaStream_t stream = side == 0 ? stream_a : stream_b;
        check_cu(tp_link_wait_value64()((CUstream)stream,
                                        reinterpret_cast<CUdeviceptr>(peer_release_(side, parity)),
                                        seq, kTpLinkWaitGeq),
                 "TpLink cuStreamWaitValue64");
        tp_link_allreduce_argmax_consume<<<1, kBlockThreads, 0, stream>>>(
            candidate_(side, parity), candidate_(1 - side, parity),
            side == 0 ? out_token_a : out_token_b, peer_release_(side, parity));
        CUDA_CHECK(cudaGetLastError());
    }
}

void TpLink::allreduce_add_side(int side, cudaStream_t stream, __nv_bfloat16* local,
                                std::size_t n) {
    allreduce_add_side(side, stream, local, local, n, 0);
}

void TpLink::allreduce_add_side(int side, cudaStream_t stream, const __nv_bfloat16* src,
                                __nv_bfloat16* dst, std::size_t n, int clear_src) {
    if (side != 0 && side != 1) {
        throw std::invalid_argument("TpLink side launch requires side 0 or 1");
    }
    if (n == 0 || n > max_elements_) {
        throw std::invalid_argument("TpLink allreduce_add n must be in (0, max_elements]");
    }
    if (clear_src != 0 && src == dst) {
        throw std::invalid_argument("TpLink allreduce_add clear_src requires distinct src/dst");
    }
    const std::uint64_t seq   = ++side_seq_[side];
    const int parity          = static_cast<int>(seq & 1u);
    const unsigned int blocks = exchange_blocks(n);
    const TpLinkSync sync     = sync_for_(side, seq);

    CUDA_CHECK(cudaSetDevice(ranks_[side]));
    if (mode_ == TpLinkMode::Spin) {
        // src is only written back when clear_src re-arms the caller's scratch accumulator
        // (the host signature keeps src const for every ordinary in-place caller).
        tp_link_allreduce_add_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            const_cast<__nv_bfloat16*>(src), dst, n, payload_(side, parity),
            payload_(1 - side, parity), sync, clear_src);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // The gate waits for the peer's mirrored publish, which the peer's own schedule call
    // enqueues (the lead thread runs both passes); neither side's publish blocks, so the
    // pair always completes.
    tp_link_allreduce_add_publish<<<blocks, kBlockThreads, 0, stream>>>(
        src, n, payload_(side, parity), sync);
    CUDA_CHECK(cudaGetLastError());
    check_cu(tp_link_wait_value64()((CUstream)stream,
                                    reinterpret_cast<CUdeviceptr>(peer_release_(side, parity)),
                                    seq, kTpLinkWaitGeq),
             "TpLink cuStreamWaitValue64");
    tp_link_allreduce_add_consume<<<blocks, kBlockThreads, 0, stream>>>(
        src, dst, n, payload_(1 - side, parity), peer_release_(side, parity));
    CUDA_CHECK(cudaGetLastError());
}

void TpLink::allreduce_argmax_side(int side, cudaStream_t stream, const __nv_bfloat16* logits,
                                   std::size_t n, std::int32_t index_base,
                                   std::int32_t* out_token) {
    if (side != 0 && side != 1) {
        throw std::invalid_argument("TpLink side launch requires side 0 or 1");
    }
    if (n == 0 || n > max_elements_) {
        throw std::invalid_argument("TpLink allreduce_argmax slice exceeds max_elements");
    }
    const std::uint64_t seq = ++side_seq_[side];
    const int parity        = static_cast<int>(seq & 1u);
    const TpLinkSync sync   = sync_for_(side, seq);

    CUDA_CHECK(cudaSetDevice(ranks_[side]));
    if (mode_ == TpLinkMode::Spin) {
        tp_link_allreduce_argmax_kernel<<<1, kBlockThreads, 0, stream>>>(
            logits, n, index_base, out_token, candidate_(side, parity), candidate_(1 - side, parity),
            sync);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    tp_link_allreduce_argmax_publish<<<1, kBlockThreads, 0, stream>>>(
        logits, n, index_base, candidate_(side, parity), sync);
    CUDA_CHECK(cudaGetLastError());
    check_cu(tp_link_wait_value64()((CUstream)stream,
                                    reinterpret_cast<CUdeviceptr>(peer_release_(side, parity)),
                                    seq, kTpLinkWaitGeq),
             "TpLink cuStreamWaitValue64");
    tp_link_allreduce_argmax_consume<<<1, kBlockThreads, 0, stream>>>(
        candidate_(side, parity), candidate_(1 - side, parity), out_token,
        peer_release_(side, parity));
    CUDA_CHECK(cudaGetLastError());
}

void TpLink::allreduce_argmax_rows_side(int side, cudaStream_t stream, const __nv_bfloat16* logits,
                                        std::size_t n, std::size_t stride, std::int32_t rows,
                                        std::int32_t index_base, const std::int32_t* id_map,
                                        std::int32_t* out_tokens) {
    if (side != 0 && side != 1) {
        throw std::invalid_argument("TpLink side launch requires side 0 or 1");
    }
    if (rows <= 0 || n == 0 || n > max_elements_ ||
        static_cast<std::size_t>(rows) * stride < static_cast<std::size_t>(rows) * n) {
        throw std::invalid_argument("TpLink allreduce_argmax_rows shape exceeds max_elements");
    }
    if (mode_ != TpLinkMode::Spin) {
        // The row winners reduce inside the rendezvous kernel; a stream-wait split would
        // need per-row publish/consume variants. TP schedules construct the link in Spin.
        throw std::invalid_argument("TpLink allreduce_argmax_rows requires the Spin exchange");
    }
    const std::uint64_t seq = ++side_seq_[side];
    const int parity        = static_cast<int>(seq & 1u);
    const TpLinkSync sync   = sync_for_(side, seq);

    CUDA_CHECK(cudaSetDevice(ranks_[side]));
    tp_link_allreduce_argmax_rows_kernel<<<static_cast<unsigned int>(rows), kBlockThreads, 0,
                                           stream>>>(logits, n, stride, rows, index_base, id_map,
                                                     out_tokens, candidate_(side, parity),
                                                     candidate_(1 - side, parity), sync);
    CUDA_CHECK(cudaGetLastError());
}

void TpLink::allgather(cudaStream_t stream_a, cudaStream_t stream_b, const __nv_bfloat16* src_a,
                       const __nv_bfloat16* src_b, __nv_bfloat16* dst_a, __nv_bfloat16* dst_b,
                       std::size_t n_a, std::size_t n_b) {
    if (n_a > max_elements_ || n_b > max_elements_) {
        throw std::invalid_argument("TpLink allgather src exceeds max_elements");
    }
    const std::uint64_t seq      = ++seq_;
    const int parity             = static_cast<int>(seq & 1u);
    const unsigned int blocks    = exchange_blocks(n_a + n_b);
    const TpLinkSync sync_a      = sync_for_(0, seq);
    const TpLinkSync sync_b      = sync_for_(1, seq);

    if (mode_ == TpLinkMode::Spin) {
        CUDA_CHECK(cudaSetDevice(ranks_[0]));
        tp_link_allgather_kernel<<<blocks, kBlockThreads, 0, stream_a>>>(
            src_a, dst_a, n_a, 0, n_b, n_a, payload_(0, parity), payload_(1, parity), sync_a);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaSetDevice(ranks_[1]));
        tp_link_allgather_kernel<<<blocks, kBlockThreads, 0, stream_b>>>(
            src_b, dst_b, n_b, n_a, n_a, 0, payload_(1, parity), payload_(0, parity), sync_b);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(cudaSetDevice(ranks_[0]));
    tp_link_allgather_publish<<<blocks, kBlockThreads, 0, stream_a>>>(src_a, n_a, payload_(0, parity),
                                                                      sync_a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(ranks_[1]));
    tp_link_allgather_publish<<<blocks, kBlockThreads, 0, stream_b>>>(src_b, n_b, payload_(1, parity),
                                                                      sync_b);
    CUDA_CHECK(cudaGetLastError());

    for (int side = 0; side < 2; ++side) {
        CUDA_CHECK(cudaSetDevice(ranks_[side]));
        const __nv_bfloat16* src         = side == 0 ? src_a : src_b;
        __nv_bfloat16* dst               = side == 0 ? dst_a : dst_b;
        const std::size_t n_local        = side == 0 ? n_a : n_b;
        const std::size_t local_offset   = side == 0 ? 0 : n_a;
        const std::size_t n_peer         = side == 0 ? n_b : n_a;
        const std::size_t peer_offset    = side == 0 ? n_a : 0;
        cudaStream_t stream              = side == 0 ? stream_a : stream_b;
        check_cu(tp_link_wait_value64()((CUstream)stream,
                                        reinterpret_cast<CUdeviceptr>(peer_release_(side, parity)),
                                        seq, kTpLinkWaitGeq),
                 "TpLink cuStreamWaitValue64");
        tp_link_allgather_consume<<<blocks, kBlockThreads, 0, stream>>>(
            src, dst, n_local, local_offset, n_peer, peer_offset, payload_(1 - side, parity),
            peer_release_(side, parity));
        CUDA_CHECK(cudaGetLastError());
    }
}

void TpLink::reset(cudaStream_t stream_a, cudaStream_t stream_b) {
    const cudaStream_t streams[2] = {stream_a, stream_b};
    for (int side = 0; side < 2; ++side) {
        CUDA_CHECK(cudaSetDevice(ranks_[side]));
        CUDA_CHECK(cudaMemsetAsync(slots_[side], 0, sizeof(TpLinkDeviceSlot) * 2, streams[side]));
    }
    reset_control_(); // host write; valid only because both ranks are idle here
    seq_          = 0;
    side_seq_[0]  = 0;
    side_seq_[1]  = 0;
}

TpLinkHealth TpLink::check_health() const noexcept {
    TpLinkHealth health;
    health.error[0] = control_->error[0];
    health.error[1] = control_->error[1];
    for (int side = 0; side < 2; ++side) {
        for (int parity = 0; parity < 2; ++parity) {
            health.release[side][parity] = control_->release[side][parity];
        }
        for (int word = 0; word < 4; ++word) {
            health.diag[side][word] = control_->diag[side][word];
        }
    }
    return health;
}

// ---- WDDM clock holder ---------------------------------------------------------
//
// Exchange-gated decode leaves the gate-idle rank's stream parked at every stream-wait
// gate, and the Windows clock governor reacts by parking that rank's SM clocks (measured
// during M4a graph rounds: RTX 4080 at 495 MHz / 21% util while its peer held
// 1980 MHz / 99%), which the per-exchange lockstep couples into the whole round. The
// holder keeps the utilization signal continuous with short spinner kernels pulsed from
// a host thread (TpProgram): 200-token steady state 14.5 -> 16.4 tok/s. Two measured
// constraints shape it: a NEVER-ENDING kernel wedges the WDDM queue (every later
// submission in the context queues behind it — full deadlock), and a separate-process
// warmer does not work at all (WDDM time-slices processes; it slowed rounds
// 71 -> 106 ms). Each pulse costs one SM slot and no memory traffic.

TpLinkSync TpLink::sync_for_(int side, std::uint64_t seq) const noexcept {
    return TpLinkSync{reinterpret_cast<TpLinkControl*>(device_base_[side] + payload_stride_ * 4),
                      side, static_cast<int>(seq & 1u), seq, slots_[side]};
}

volatile __nv_bfloat16* TpLink::payload_(int side, int parity) const noexcept {
    const std::size_t slice = static_cast<std::size_t>(side * 2 + parity);
    return reinterpret_cast<volatile __nv_bfloat16*>(device_base_[side] + slice * payload_stride_);
}

volatile TpLinkArgmaxCandidate* TpLink::candidate_(int side, int parity) const noexcept {
    return reinterpret_cast<volatile TpLinkArgmaxCandidate*>(payload_(side, parity));
}

volatile std::uint64_t* TpLink::peer_release_(int side, int parity) const noexcept {
    TpLinkControl* mapped =
        reinterpret_cast<TpLinkControl*>(device_base_[side] + payload_stride_ * 4);
    return &mapped->release[1 - side][parity];
}

void TpLink::reset_control_() noexcept {
    for (int side = 0; side < 2; ++side) {
        for (int parity = 0; parity < 2; ++parity) { control_->release[side][parity] = 0; }
        control_->error[side] = 0;
        for (int word = 0; word < 4; ++word) { control_->diag[side][word] = 0; }
    }
}

__global__ void tp_clock_holder_kernel(int iterations) {
    float x = 1.0F;
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll 8
        for (int j = 0; j < 32; ++j) { x = x * 0.9999F + 0.0001F; }
        if ((i & 63) == 63) { __nanosleep(1000); }
    }
    if (x == 42.0F) { printf(""); } // never taken; keeps the arithmetic live
}

void tp_clock_holder_pulse(cudaStream_t holder_stream, int iterations) {
    tp_clock_holder_kernel<<<1, 64, 0, holder_stream>>>(iterations);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
