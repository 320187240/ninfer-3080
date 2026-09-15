#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.
//
// TpProgram: the two-rank tensor-parallel coordinator. It owns one per-rank ProgramImpl of a
// shard-shaped Variant instantiation (each with its own DeviceContext, arenas, and mirror of
// the request/lane state) plus the TpLink transport, and exposes the ProgramImplCore surface
// the ConcurrentExecutor drives. Execution methods enqueue both ranks' mirrored schedules
// CONCURRENTLY, this rank on the calling thread and the peer on a worker thread: every
// TpLink exchange blocks a rank's stream until the peer's publish for that exchange runs
// (the stream-wait gate or the in-kernel spin alike), so overlapping the two enqueue passes
// is what keeps the WDDM software queues from filling behind an exchange whose peer work has
// not been enqueued yet (a strictly serial enqueue of whole rounds deadlocks there).
// Planning and lifecycle state stay mirrored on both ranks because every observable
// decision (greedy tokens, lane lifecycle) is deterministic and identical.

#include "core/device.h"
#include "core/tp_link.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/tp_exec.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

template <class VariantT>
class TpProgram final {
public:
    using Impl         = qwen3_6::detail::ProgramImpl<VariantT>;
    using TextConfig   = typename VariantT::TextConfig;
    using SequencePlan = qwen3_6::SequencePlan<VariantT>;
    using RequestBasePlanImpl =
        qwen3_6::detail::RequestBasePlanImpl<VariantT>;
    using RequestPlanImpl = qwen3_6::detail::RequestPlanImpl<VariantT>;

    // Clock-holder cadence (M4a). Widening the pulse footprint was measured and rejected:
    // see core/tp_link.h for the M4c negative result.
    static constexpr int kClockHolderIterations = 524288;
    static constexpr int kClockHolderSleepMs    = 4;

    TpProgram(const typename VariantT::ModelView& model_a, const typename VariantT::ModelView& model_b,
              SequencePlan&& plan, DeviceContext& device_a, DeviceContext& device_b,
              int primary_device, TpClockHolderMode clock_holder = TpClockHolderMode::Demand,
              std::uint32_t clock_holder_hold_ms = 10'000)
        : device_a_(device_a), device_b_(device_b), primary_device_(primary_device),
          prefill_chunk_(plan.impl_ != nullptr ? plan.impl_->prefill_chunk : 0),
          hidden_(TextConfig::hidden), vocab_rows_(TextConfig::output_rows),
          max_exchange_elements_(static_cast<std::size_t>(prefill_chunk_) * hidden_) {
        if (plan.impl_ == nullptr) { throw std::invalid_argument("TP sequence plan is empty"); }
        const std::size_t peer_transient_bytes = plan.request_transient_capacity_bytes();
        if (device_a.device == device_b.device) {
            throw std::invalid_argument("TP ranks must use two distinct devices");
        }
        CUDA_CHECK(cudaSetDevice(device_a.device));
        programs_[0] = std::make_unique<Impl>(model_a, *plan.impl_, device_a);
        CUDA_CHECK(cudaSetDevice(device_b.device));
        programs_[1] = std::make_unique<Impl>(model_b, *plan.impl_, device_b);
        plan.impl_.reset();

        // Spin: one in-kernel rendezvous per exchange (publish, bounded-spin on the peer's
        // mapped release, consume) with no channel-scheduler gate transition between the
        // kernels; the go broadcast is replay-stable under captured schedules (equality
        // gate, see core/tp_link.cuh). StreamWait remains available through the mode param.
        link_ = std::make_unique<TpLink>(device_a.device, device_b.device,
                                         max_exchange_elements_, TpLinkMode::Spin);
        for (int side = 0; side < 2; ++side) {
            DeviceContext& ctx = side == 0 ? device_a_ : device_b_;
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaEventCreateWithFlags(&flush_events_[side], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&exec_[side].local_ids),
                                  static_cast<std::size_t>(prefill_chunk_) * sizeof(std::int32_t)));
        }
        CUDA_CHECK(cudaSetDevice(device_b.device));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&zero_residual_),
                              static_cast<std::size_t>(prefill_chunk_) *
                                  static_cast<std::size_t>(hidden_) * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(zero_residual_, 0,
                              static_cast<std::size_t>(prefill_chunk_) *
                                  static_cast<std::size_t>(hidden_) * sizeof(__nv_bfloat16)));
        if (peer_transient_bytes != 0) {
            // Rank 1's mirror of the request transient (the engine's RequestMemory lives on
            // the primary device): a vision item encodes into rank 0's transient and lands
            // here bit-identically through the TpLink broadcast, so the mirrored text
            // prefill scatters from rank-1-local memory. The bytes were reserved by rank 1's
            // own sequence-plan preflight, exactly like rank 0's copy.
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&peer_transient_),
                                  peer_transient_bytes));
        }

        for (int side = 0; side < 2; ++side) {
            exec_[side].side          = side;
            exec_[side].link          = link_.get();
            exec_[side].hidden        = hidden_;
            exec_[side].vocab_rows    = vocab_rows_;
            exec_[side].vocab_base    = static_cast<std::int64_t>(side) * vocab_rows_;
            exec_[side].max_tokens    = static_cast<std::int32_t>(prefill_chunk_);
            exec_[side].zero_residual = side == 0 ? nullptr : zero_residual_;
            programs_[side]->set_tp_execution(&exec_[side]);
        }
        programs_[0]->set_round_sync(&TpProgram::lead_sync_thunk_, this);
        programs_[1]->set_round_sync(&TpProgram::follow_sync_thunk_, this);

        // Graph preparation is deferred to here: its code warm-up and per-topology
        // qualification launches execute the real TP schedule, whose every exchange
        // rendezvous with the peer rank's mirror. Both ranks therefore prepare concurrently
        // on their own threads, exactly like a decode round; a per-rank sequential prepare
        // would hang on the first exchange gate (the stream wait has no timeout).
        std::exception_ptr peer_error;
        std::thread prepare_peer([&peer_error, this] {
            try {
                CUDA_CHECK(cudaSetDevice(device_b_.device));
                programs_[1]->prepare_program_graphs();
            } catch (...) { peer_error = std::current_exception(); }
        });
        std::exception_ptr lead_error;
        try {
            CUDA_CHECK(cudaSetDevice(device_a_.device));
            programs_[0]->prepare_program_graphs();
        } catch (...) { lead_error = std::current_exception(); }
        prepare_peer.join();
        if (lead_error != nullptr) { std::rethrow_exception(lead_error); }
        if (peer_error != nullptr) { std::rethrow_exception(peer_error); }

        // Exchange-gated rounds leave each rank's stream waiting at every exchange (parked
        // at a stream-wait gate, or resident in a spin kernel between compute kernels),
        // which makes the Windows clock governor park the gate-idle rank's SM clocks
        // (measured during M4a graph rounds: RTX 4080 at 495 MHz / 21% util while its peer
        // held 1980 MHz / 99%), and the lockstep couples that parking into the whole round.
        // Each rank pulses short clock-holder kernels from its own thread; the spinners
        // cost one SM slot, no memory traffic, and always terminate (a never-ending kernel
        // wedges the WDDM queue — every later submission in the context queues behind it).
        //
    // Demand mode: pulses run only while engine work is recent — every mirrored
    // lifecycle method refreshes the activity deadline — and stop holder_hold_ms_ after
    // the last op, so an idle server parks its clocks and saves power. The grace is a
    // measured quantity, not a guess: a real agent session's inter-request gaps (n=362)
    // have median 607 ms / p75 1.2 s / p90 3.9 s, so a 500 ms window expired before 58%
    // of requests arrived and demand behaved like off; 10 s covers 93% while
    // minutes-scale pauses still park. Pulses HOLD clocks that are up but cannot RAISE
    // parked ones (a 1-block pulse is ~1% util; always-on holds rank 0 at only ~780 MHz
    // for the same reason), so the grace must bridge the whole gap distribution you
    // care about. Real agentic sessions measurably need the holder while requests flow
    // (~/ninfer.jsonl: median 59.2 tok/s fully-off), while an always-on holder burns
    // ~97 W at idle for nothing. Always restores the original behaviour; Off disables.
    // NINFER_TP_NO_CLOCK_HOLDER=1 forces off and NINFER_TP_CLOCK_HOLDER_ALWAYS=1 forces
    // always, overriding the requested mode for diagnostics.
        TpClockHolderMode effective_holder = clock_holder;
        if (std::getenv("NINFER_TP_NO_CLOCK_HOLDER") != nullptr) {
            effective_holder = TpClockHolderMode::Off;
        } else if (std::getenv("NINFER_TP_CLOCK_HOLDER_ALWAYS") != nullptr) {
            effective_holder = TpClockHolderMode::Always;
        }
        clock_run_.store(effective_holder != TpClockHolderMode::Off);
        holder_always_ = effective_holder == TpClockHolderMode::Always;
        holder_hold_ms_ =
            clock_holder_hold_ms == 0 ? 10'000 : clock_holder_hold_ms;
        for (int side = 0; side < 2 && clock_run_.load(); ++side) {
            DeviceContext& ctx = side == 0 ? device_a_ : device_b_;
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaStreamCreateWithFlags(&clock_streams_[side], cudaStreamNonBlocking));
            clock_threads_[side] = std::thread([this, side] {
                CUDA_CHECK(cudaSetDevice(device_(side).device));
                while (clock_run_.load()) {
                    if (holder_always_ ||
                        holder_until_ms_.load(std::memory_order_acquire) > holder_now_ms_()) {
                        tp_clock_holder_pulse(clock_streams_[side], kClockHolderIterations);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(kClockHolderSleepMs));
                }
            });
        }
        CUDA_CHECK(cudaSetDevice(primary_device_));
    }

    ~TpProgram() noexcept {
        clock_run_.store(false);
        for (int side = 0; side < 2; ++side) {
            if (clock_threads_[side].joinable()) { clock_threads_[side].join(); }
        }
        try {
            for (int side = 0; side < 2; ++side) {
                DeviceContext& ctx = side == 0 ? device_a_ : device_b_;
                CUDA_CHECK(cudaSetDevice(ctx.device));
                if (clock_streams_[side] != nullptr) {
                    cudaStreamSynchronize(clock_streams_[side]);
                    cudaStreamDestroy(clock_streams_[side]);
                    clock_streams_[side] = nullptr;
                }
                if (exec_[side].local_ids != nullptr) {
                    cudaFree(exec_[side].local_ids);
                    exec_[side].local_ids = nullptr;
                }
                if (flush_events_[side] != nullptr) {
                    cudaEventDestroy(flush_events_[side]);
                    flush_events_[side] = nullptr;
                }
            }
            CUDA_CHECK(cudaSetDevice(device_b_.device));
            if (zero_residual_ != nullptr) {
                cudaFree(zero_residual_);
                zero_residual_ = nullptr;
            }
            if (peer_transient_ != nullptr) {
                cudaFree(peer_transient_);
                peer_transient_ = nullptr;
            }
        } catch (...) {}
        try {
            CUDA_CHECK(cudaSetDevice(primary_device_));
        } catch (...) {}
    }

    TpProgram(const TpProgram&)            = delete;
    TpProgram& operator=(const TpProgram&) = delete;
    TpProgram(TpProgram&&)                 = delete;
    TpProgram& operator=(TpProgram&&)      = delete;

    [[nodiscard]] qwen3_6::RequestBasePlan<VariantT>
    plan_request_base(const qwen3_6::PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options) {
        // TP mode restrictions are request-visible: sampling must be greedy (the
        // vocab-parallel argmax reduce is the sampling path). Repetition penalties are part of
        // ops::sample's full-vocab path, which TP eager greedy bypasses, so they are forced
        // off rather than silently ignored (an M4 gap to close with the sampling reduce).
        // Prefix reuse is on by default in TP serving (opt out with --no-tp-prefix-reuse →
        // ExecutionOptions::tp_prefix_reuse); the retention coordinator validates both ranks'
        // mirrored SequenceState on every lifecycle op (see tp_retention.h), so a plain
        // allow_prefix_reuse=true from the server unlocks it here.
        runtime::ResolvedExecutionOptions tp_options = options;
        tp_options.allow_prefix_reuse =
            tp_options.allow_prefix_reuse && tp_options.tp_prefix_reuse;
        if (tp_options.sampling.temperature > 0.0F) {
            throw RequestError(RequestErrorKind::Unavailable,
                               "two-GPU tensor-parallel mode requires greedy sampling "
                               "(temperature 0)");
        }
        tp_options.sampling.presence_penalty  = 0.0F;
        tp_options.sampling.frequency_penalty = 0.0F;
        return programs_[0]->plan_request_base(qwen3_6::PreparedPromptAccess::view(prompt),
                                               tp_options);
    }

    [[nodiscard]] qwen3_6::RequestPlan<VariantT>
    plan_request_for_lane(std::uint32_t lane, const qwen3_6::PreparedPrompt& prompt,
                          const qwen3_6::RequestBasePlan<VariantT>& base) {
        // Planning must observe exactly the state the coordinator last validated; a lead-rank
        // drift means an unmirrored mutation happened and reuse decisions would be unsafe.
        retention_.check_fault();
        retention_.validate_lead(lane, programs_[0]->sequence_retention_digest(lane));
        return programs_[0]->plan_request_for_lane(lane, qwen3_6::PreparedPromptAccess::view(prompt),
                                                   base);
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const qwen3_6::RequestPlan<VariantT>& plan) const noexcept {
        return programs_[0]->can_admit_lane(lane, plan) && programs_[1]->can_admit_lane(lane, plan);
    }

    [[nodiscard]] bool can_admit_lane_after_retained_eviction(
        std::uint32_t lane, const qwen3_6::RequestPlan<VariantT>& plan) const noexcept {
        return programs_[0]->can_admit_lane_after_retained_eviction(lane, plan) &&
               programs_[1]->can_admit_lane_after_retained_eviction(lane, plan);
    }

    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept {
        const runtime::AdmissionResources a = programs_[0]->admission_capacity();
        const runtime::AdmissionResources b = programs_[1]->admission_capacity();
        return runtime::AdmissionResources{
            .active_lanes     = std::min(a.active_lanes, b.active_lanes),
            .main_kv_pages    = std::min(a.main_kv_pages, b.main_kv_pages),
            .backend_kv_pages = std::min(a.backend_kv_pages, b.backend_kv_pages),
        };
    }

    [[nodiscard]] runtime::PrefillStepResult
    start_prefill_lane(std::uint32_t lane, qwen3_6::PreparedPrompt&& prompt,
                       qwen3_6::RequestPlan<VariantT>&& plan, runtime::TransientRegion transient) {
        if (plan.impl_ == nullptr) { throw std::invalid_argument("TP request plan is empty"); }
        retention_.check_fault();
        note_holder_activity_();
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        peer_op_     = PeerOp::StartPrefill;
        peer_lane_   = lane;
        peer_prompt_ = qwen3_6::PreparedPromptAccess::view(prompt); // mirror for the peer rank
        peer_plan_   = std::make_unique<RequestPlanImpl>(*plan.impl_);
        peer_transient = transient;
        if (peer_transient_ != nullptr) {
            // The peer's transient mirror swaps in rank 1's device buffer; size and
            // alignment are the same activation the engine performed for rank 0.
            peer_transient.data = reinterpret_cast<std::byte*>(peer_transient_);
        }
        std::exception_ptr peer_error;
        std::thread peer([&peer_error, this] {
            try { run_peer_(); } catch (...) { peer_error = std::current_exception(); }
        });
        runtime::PrefillStepResult result;
        try {
            result = programs_[0]->start_prefill_lane(
                lane, qwen3_6::PreparedPromptAccess::take(std::move(prompt)), std::move(plan),
                transient);
        } catch (...) {
            peer.join();
            peer_op_ = PeerOp::None;
            retention_.invalidate_all();
            throw;
        }
        peer.join();
        peer_op_ = PeerOp::None;
        if (peer_error != nullptr) {
            retention_.invalidate_all();
            std::rethrow_exception(peer_error);
        }
        validate_retention_(lane);
        return result;
    }

    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        retention_.check_fault();
        note_holder_activity_();
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        peer_op_   = PeerOp::AdvancePrefill;
        peer_lane_ = lane;
        std::exception_ptr peer_error;
        std::thread peer([&peer_error, this] {
            try { run_peer_(); } catch (...) { peer_error = std::current_exception(); }
        });
        runtime::PrefillStepResult result;
        try {
            result = programs_[0]->advance_prefill_lane(lane);
        } catch (...) {
            peer.join();
            peer_op_ = PeerOp::None;
            throw;
        }
        peer.join();
        peer_op_ = PeerOp::None;
        if (peer_error != nullptr) { std::rethrow_exception(peer_error); }
        return result;
    }

    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets) {
        retention_.check_fault();
        note_holder_activity_();
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        peer_op_ = PeerOp::Decode;
        peer_lanes_.assign(lanes.begin(), lanes.end());
        peer_budgets_.assign(budgets.begin(), budgets.end());
        std::exception_ptr peer_error;
        std::thread peer([&peer_error, this] {
            try { run_peer_(); } catch (...) { peer_error = std::current_exception(); }
        });
        runtime::BatchedGeneratedRound round;
        try {
            round = programs_[0]->decode_batch(lanes, budgets);
        } catch (...) {
            peer.join();
            peer_op_ = PeerOp::None;
            throw;
        }
        peer.join();
        peer_op_ = PeerOp::None;
        if (peer_error != nullptr) { std::rethrow_exception(peer_error); }
        return round;
    }

    void resolve_prefill_lane(std::uint32_t lane, bool terminal) {
        note_holder_activity_();
        programs_[0]->resolve_prefill_lane(lane, terminal);
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        programs_[1]->resolve_prefill_lane(lane, terminal);
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        validate_retention_(lane);
    }

    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled) {
        note_holder_activity_();
        programs_[0]->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        programs_[1]->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        for (const std::uint32_t lane : lanes) { validate_retention_(lane); }
    }

    void abort_lane(std::uint32_t lane) noexcept {
        // Mirrored on both ranks before digesting (see tp_retention.h): both ranks execute the
        // same deterministic state machine on the same lane, so abort retention decisions
        // (retain at the completed-chunk boundary vs discard) agree and the recorded digests
        // describe one coherent retained prefix. A divergence would surface as a sticky fault
        // at the next throwing entry point.
        programs_[0]->abort_lane(lane);
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        programs_[1]->abort_lane(lane);
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        record_retention_(lane);
    }

    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept {
        return programs_[0]->has_retained_lane(lane) && programs_[1]->has_retained_lane(lane);
    }

    void evict_retained_lane(std::uint32_t lane) noexcept {
        programs_[0]->evict_retained_lane(lane);
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        programs_[1]->evict_retained_lane(lane);
        CUDA_CHECK(cudaSetDevice(device_a_.device));
        record_retention_(lane);
    }

    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept {
        return programs_[0]->generation_timings_lane(lane);
    }

    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept {
        return programs_[0]->speculative_stats_lane(lane);
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        MemorySummary a = programs_[0]->memory_summary();
        const MemorySummary b = programs_[1]->memory_summary();
        const auto sum = [](const ArenaMemorySummary& x,
                            const ArenaMemorySummary& y) noexcept {
            return ArenaMemorySummary{x.capacity_bytes + y.capacity_bytes,
                                       x.used_bytes + y.used_bytes,
                                       x.peak_used_bytes + y.peak_used_bytes};
        };
        a.weights   = sum(a.weights, b.weights);
        a.sequence  = sum(a.sequence, b.sequence);
        a.workspace = sum(a.workspace, b.workspace);
        a.kv_payload_bytes += b.kv_payload_bytes;
        a.tp_devices       = {a.device, b.device};
        return a;
    }

    void reset_memory_peaks() noexcept {
        programs_[0]->reset_memory_peaks();
        programs_[1]->reset_memory_peaks();
    }

private:
    enum class PeerOp : std::uint8_t {
        None,
        StartPrefill,
        AdvancePrefill,
        Decode,
    };

    static void lead_sync_thunk_(void* context) { static_cast<TpProgram*>(context)->lead_sync_(); }
    static void follow_sync_thunk_(void* context) {
        static_cast<TpProgram*>(context)->follow_sync_();
    }

    [[nodiscard]] DeviceContext& device_(int side) noexcept {
        return side == 0 ? device_a_ : device_b_;
    }

    // Forces WDDM to submit this rank's enqueued schedule before the host blocks on the peer;
    // the peer's exchange kernels spin until these kernels run.
    void flush_(int side) {
        DeviceContext& ctx = device_(side);
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaEventRecord(flush_events_[side], ctx.stream));
        (void)cudaEventQuery(flush_events_[side]);
    }

    void lead_sync_() {
        // The peer's mirrored schedule runs concurrently on its worker thread (see the
        // execution methods), so this seam only has to flush and drain this rank: the
        // event queries force WDDM to submit the enqueued exchanges, and the stream poll
        // drives both queues while the round's gates clear as the peer publishes.
        flush_(0);
        flush_(1);
        // Poll both streams to completion: the poll keeps both WDDM queues actively submitted
        // instead of blocking the host on one rank while the other's pending launches wait for
        // submission behind a full software queue.
        bool pending_a = true;
        bool pending_b = true;
        while (pending_a || pending_b) {
            if (pending_a) {
                CUDA_CHECK(cudaSetDevice(device_a_.device));
                const cudaError_t state = cudaStreamQuery(device_a_.stream);
                if (state != cudaErrorNotReady) {
                    CUDA_CHECK(state);
                    pending_a = false;
                }
            }
            if (pending_b) {
                CUDA_CHECK(cudaSetDevice(device_b_.device));
                const cudaError_t state = cudaStreamQuery(device_b_.stream);
                if (state != cudaErrorNotReady) {
                    CUDA_CHECK(state);
                    pending_b = false;
                }
            }
        }
        const TpLinkHealth health = link_->check_health();
        if (!health.ok()) {
            // Only the Spin mode's bounded give-up ever writes the error words now (the
            // stream-wait gate has no timeout); a divergence there hangs the stream instead
            // of consuming stale data, so this warning always names a real transport event.
            std::fprintf(stderr,
                         "ninfer tp warning: exchange spin timed out (side0 error=%llu seq=%llu, "
                         "side1 error=%llu seq=%llu, release a0/a1/b0/b1=%llu/%llu/%llu/%llu, "
                         "diag0 seq/cyc/peer/own=%llu/%llu/%llu/%llu, "
                         "diag1 seq/cyc/peer/own=%llu/%llu/%llu/%llu)\n",
                         static_cast<unsigned long long>(health.error[0]),
                         static_cast<unsigned long long>(link_->side_seq(0)),
                         static_cast<unsigned long long>(health.error[1]),
                         static_cast<unsigned long long>(link_->side_seq(1)),
                         static_cast<unsigned long long>(health.release[0][0]),
                         static_cast<unsigned long long>(health.release[0][1]),
                         static_cast<unsigned long long>(health.release[1][0]),
                         static_cast<unsigned long long>(health.release[1][1]),
                         static_cast<unsigned long long>(health.diag[0][0]),
                         static_cast<unsigned long long>(health.diag[0][1]),
                         static_cast<unsigned long long>(health.diag[0][2]),
                         static_cast<unsigned long long>(health.diag[0][3]),
                         static_cast<unsigned long long>(health.diag[1][0]),
                         static_cast<unsigned long long>(health.diag[1][1]),
                         static_cast<unsigned long long>(health.diag[1][2]),
                         static_cast<unsigned long long>(health.diag[1][3]));
        }
        CUDA_CHECK(cudaSetDevice(primary_device_));
    }

    void follow_sync_() {
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        device_b_.synchronize();
    }

    void run_peer_() {
        CUDA_CHECK(cudaSetDevice(device_b_.device));
        switch (peer_op_) {
        case PeerOp::StartPrefill: {
            qwen3_6::RequestPlan<VariantT> plan{
                std::unique_ptr<RequestPlanImpl>(std::move(peer_plan_))};
            // The peer's return values mirror rank 0's deterministic results; the lead
            // thread's copy is the one observed.
            (void)programs_[1]->start_prefill_lane(peer_lane_, std::move(peer_prompt_),
                                                   std::move(plan), peer_transient);
            break;
        }
        case PeerOp::AdvancePrefill:
            (void)programs_[1]->advance_prefill_lane(peer_lane_);
            break;
        case PeerOp::Decode:
            (void)programs_[1]->decode_batch(peer_lanes_, peer_budgets_);
            break;
        case PeerOp::None:
            throw std::logic_error("TP peer schedule has no pending operation");
        }
    }

    // Retention coherence: after every mirrored lifecycle op, the lead rank's digest becomes
    // the expectation and the peer's copy must match it exactly (field-level diff on throw).
    void validate_retention_(std::uint32_t lane) {
        retention_.record_lead(lane, programs_[0]->sequence_retention_digest(lane));
        retention_.validate_peer(lane, programs_[1]->sequence_retention_digest(lane));
        retention_.note_op(lane, "validate");
    }

    // noexcept mirror for abort/evict: observe only; a divergence is stored and rethrown by
    // check_fault() at the next throwing entry point.
    void record_retention_(std::uint32_t lane) noexcept {
        try {
            retention_.record_lead(lane, programs_[0]->sequence_retention_digest(lane));
            retention_.observe_peer(lane, programs_[1]->sequence_retention_digest(lane));
        } catch (...) {
            retention_.invalidate_all();
        }
    }

    DeviceContext& device_a_;
    DeviceContext& device_b_;
    const int primary_device_;
    qwen3_6::detail::TpRetentionCoordinator retention_{kMaximumConcurrency};
    const std::uint32_t prefill_chunk_;
    const std::int32_t hidden_;
    const std::int32_t vocab_rows_;
    const std::size_t max_exchange_elements_;
    std::unique_ptr<Impl> programs_[2]{};
    std::unique_ptr<TpLink> link_;
    schedule::TpExec exec_[2]{};
    __nv_bfloat16* zero_residual_ = nullptr;
    void* peer_transient_         = nullptr;
    cudaEvent_t flush_events_[2]   = {};
    cudaStream_t clock_streams_[2] = {};
    std::thread clock_threads_[2];
    std::atomic<bool> clock_run_{false};
    // Demand-mode holder state: pulses continue while now < holder_until_ms_.
    std::atomic<long long> holder_until_ms_{0};
    std::uint32_t holder_hold_ms_ = 10'000;
    bool holder_always_ = false;

    static long long holder_now_ms_() noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void note_holder_activity_() noexcept {
        holder_until_ms_.store(holder_now_ms_() + holder_hold_ms_, std::memory_order_release);
    }

    PeerOp peer_op_ = PeerOp::None;
    std::uint32_t peer_lane_      = 0;
    qwen3_6::PreparedPromptData peer_prompt_{};
    std::unique_ptr<RequestPlanImpl> peer_plan_;
    runtime::TransientRegion peer_transient{};
    std::vector<std::uint32_t> peer_lanes_;
    std::vector<runtime::RoundBudget> peer_budgets_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
