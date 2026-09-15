#pragma once
// Retention coherence for two-rank tensor-parallel execution.
//
// The TP coordinator mirrors every ProgramImplCore lifecycle call to both ranks, so each rank's
// SequenceState (ledger, prefix identity, checkpoints, KV page allocations) evolves
// deterministically in lockstep. That property is currently true by construction; this module
// makes it CHECKABLE instead: TpRetentionCoordinator holds the lead rank's digest as the
// expectation, and any peer divergence or unexpected lead drift throws with a field-level
// diff instead of silently corrupting reused KV.
//
// Stage 1 (this file): shadow digests + validation hooks at every mirrored lifecycle op.
// Stage 2 will promote checkpoint capture / rollback / eviction into explicit coordinated
// ops and consume the op journal recorded by note_op().

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Host-side fingerprint of one lane's SequenceState. Everything that prefix-reuse planning
// reads (and everything page allocation depends on) is folded in; two ranks holding equal
// digests hold equivalent reuse state. Pure value type.
struct RetentionDigest {
    bool retained                 = false;
    bool tail_hidden_valid        = false;
    bool kv_valid                 = false;
    bool checkpoint_valid         = false;
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::uint32_t ledger_size        = 0;
    std::uint64_t ledger_hash        = 0;
    std::uint64_t identity_hash      = 0;
    std::uint64_t identity_size      = 0;
    std::int32_t rope_delta          = 0;
    std::uint32_t text_kv_valid      = 0;
    std::uint32_t mtp_kv_valid       = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::uint32_t checkpoint_frontier     = 0;
    std::uint64_t mtp_drafts_hash         = 0;
    std::uint32_t mtp_draft_count         = 0;
    std::uint32_t text_page_count         = 0;
    std::uint32_t text_page_entitlement   = 0;
    std::int32_t text_bound_row           = -1;
    std::uint64_t text_pages_hash         = 0;
    std::uint32_t backend_page_count      = 0;

    [[nodiscard]] bool operator==(const RetentionDigest&) const noexcept = default;

    // Field-by-field diff for error messages; empty when equal.
    [[nodiscard]] std::string diff(const RetentionDigest& o) const {
        auto mismatch = [](const char* name, auto a, auto b) {
            return std::string("   ") + name + ": " + std::to_string(a) + " != " +
                   std::to_string(b) + "\n";
        };
        std::string out;
        if (retained != o.retained) { out += mismatch("retained", retained, o.retained); }
        if (tail_hidden_valid != o.tail_hidden_valid) {
            out += mismatch("tail_hidden_valid", tail_hidden_valid, o.tail_hidden_valid);
        }
        if (kv_valid != o.kv_valid) { out += mismatch("kv_valid", kv_valid, o.kv_valid); }
        if (checkpoint_valid != o.checkpoint_valid) {
            out += mismatch("checkpoint_valid", checkpoint_valid, o.checkpoint_valid);
        }
        if (execution_frontier != o.execution_frontier) {
            out += mismatch("execution_frontier", execution_frontier, o.execution_frontier);
        }
        if (ledger_frontier != o.ledger_frontier) {
            out += mismatch("ledger_frontier", ledger_frontier, o.ledger_frontier);
        }
        if (ledger_size != o.ledger_size) {
            out += mismatch("ledger_size", ledger_size, o.ledger_size);
        }
        if (ledger_hash != o.ledger_hash) {
            out += mismatch("ledger_hash", ledger_hash, o.ledger_hash);
        }
        if (identity_hash != o.identity_hash) {
            out += mismatch("identity_hash", identity_hash, o.identity_hash);
        }
        if (identity_size != o.identity_size) {
            out += mismatch("identity_size", identity_size, o.identity_size);
        }
        if (rope_delta != o.rope_delta) { out += mismatch("rope_delta", rope_delta, o.rope_delta); }
        if (text_kv_valid != o.text_kv_valid) {
            out += mismatch("text_kv_valid", text_kv_valid, o.text_kv_valid);
        }
        if (mtp_kv_valid != o.mtp_kv_valid) {
            out += mismatch("mtp_kv_valid", mtp_kv_valid, o.mtp_kv_valid);
        }
        if (dflash_context_frontier != o.dflash_context_frontier) {
            out += mismatch("dflash_context_frontier", dflash_context_frontier,
                            o.dflash_context_frontier);
        }
        if (checkpoint_frontier != o.checkpoint_frontier) {
            out += mismatch("checkpoint_frontier", checkpoint_frontier, o.checkpoint_frontier);
        }
        if (mtp_drafts_hash != o.mtp_drafts_hash) {
            out += mismatch("mtp_drafts_hash", mtp_drafts_hash, o.mtp_drafts_hash);
        }
        if (mtp_draft_count != o.mtp_draft_count) {
            out += mismatch("mtp_draft_count", mtp_draft_count, o.mtp_draft_count);
        }
        if (text_page_count != o.text_page_count) {
            out += mismatch("text_page_count", text_page_count, o.text_page_count);
        }
        if (text_page_entitlement != o.text_page_entitlement) {
            out += mismatch("text_page_entitlement", text_page_entitlement,
                            o.text_page_entitlement);
        }
        if (text_bound_row != o.text_bound_row) {
            out += mismatch("text_bound_row", text_bound_row, o.text_bound_row);
        }
        if (text_pages_hash != o.text_pages_hash) {
            out += mismatch("text_pages_hash", text_pages_hash, o.text_pages_hash);
        }
        if (backend_page_count != o.backend_page_count) {
            out += mismatch("backend_page_count", backend_page_count, o.backend_page_count);
        }
        return out;
    }
};

// FNV-1a over raw bytes; shared by the digest builders.
[[nodiscard]] inline std::uint64_t retention_fnv1a(const void* data, std::size_t bytes,
                                                   std::uint64_t seed = 0xcbf29ce484222325ULL) {
    const auto* bytes8 = static_cast<const unsigned char*>(data);
    std::uint64_t hash = seed;
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= bytes8[index];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

class TpRetentionCoordinator {
public:
    explicit TpRetentionCoordinator(std::uint32_t lane_count)
        : known_(lane_count, false), expected_(lane_count), op_counts_(lane_count, 0) {}

    TpRetentionCoordinator(const TpRetentionCoordinator&)            = delete;
    TpRetentionCoordinator& operator=(const TpRetentionCoordinator&) = delete;

    // Records the lead rank's post-op digest as the new expectation for the lane.
    void record_lead(std::uint32_t lane, const RetentionDigest& lead) {
        check_lane(lane);
        expected_[lane] = lead;
        known_[lane]    = true;
    }

    // Throws when the peer rank's digest diverges from the recorded expectation.
    void validate_peer(std::uint32_t lane, const RetentionDigest& peer) const {
        check_lane(lane);
        if (!known_[lane]) { return; }
        if (expected_[lane] != peer) {
            throw std::runtime_error("TP retention divergence on lane " + std::to_string(lane) +
                                     " (rank 1 vs rank 0):\n  " +
                                     expected_[lane].diff(peer));
        }
    }

    // noexcept observation for mirroring sites that cannot throw (abort/evict): stores the
    // first divergence; check_fault() rethrows it from the next throwing entry point.
    void observe_peer(std::uint32_t lane, const RetentionDigest& peer) const noexcept {
        try {
            check_lane(lane);
            if (known_[lane] && expected_[lane] != peer && fault_.empty()) {
                fault_ = "TP retention divergence on lane " + std::to_string(lane) +
                         " (rank 1 vs rank 0):\n  " + expected_[lane].diff(peer);
            }
        } catch (...) {
            // Out-of-range lane inside a noexcept mirror cannot be reported here; the next
            // throwing validation will surface the state problem.
        }
    }

    // Rethrows a divergence previously captured by observe_peer.
    void check_fault() const {
        if (!fault_.empty()) { throw std::runtime_error(fault_); }
    }

    // Throws when the lead rank drifted from the expectation recorded at the last mirrored
    // lifecycle op (planning reads must observe exactly the coordinated state).
    void validate_lead(std::uint32_t lane, const RetentionDigest& lead) const {
        check_lane(lane);
        if (!known_[lane]) { return; }
        if (expected_[lane] != lead) {
            throw std::runtime_error("TP retention drift on lane " + std::to_string(lane) +
                                     " (rank 0 vs coordinator shadow):\n  " +
                                     expected_[lane].diff(lead));
        }
    }

    // After an exceptional path the shadow may be stale (a rank's error cleanup ran inside the
    // peer thread); suppress validation until the next successful record.
    void invalidate_all() noexcept {
        for (char& lane_known : known_) { lane_known = 0; }
    }

    // Stage-2 op journal hook: counts coordinated retention-relevant ops per lane.
    void note_op(std::uint32_t lane, const char* /*op*/) noexcept {
        if (lane < op_counts_.size()) { ++op_counts_[lane]; }
    }

    [[nodiscard]] std::uint64_t ops_recorded(std::uint32_t lane) const noexcept {
        return lane < op_counts_.size() ? op_counts_[lane] : 0;
    }

private:
    void check_lane(std::uint32_t lane) const {
        if (lane >= expected_.size()) {
            throw std::out_of_range("TpRetentionCoordinator lane " + std::to_string(lane) +
                                    " out of range");
        }
    }

    std::vector<char> known_;
    std::vector<RetentionDigest> expected_;
    std::vector<std::uint64_t> op_counts_;
    mutable std::string fault_; // first observe_peer divergence, rethrown by check_fault
};

} // namespace ninfer::targets::qwen3_6::detail
