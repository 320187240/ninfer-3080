// Pure host unit test for the TP retention coordinator: shadow digests, peer divergence
// detection, sticky faults from noexcept observe paths, and shadow invalidation. No CUDA
// devices are required.

#include "targets/qwen3_6/impl/runtime/tp_retention.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace detail = ninfer::targets::qwen3_6::detail;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename Fn>
bool throws_with(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

} // namespace

int main() {
    detail::RetentionDigest digest;
    digest.retained          = true;
    digest.ledger_size       = 128;
    digest.ledger_hash       = 0x1234567812345678ULL;
    digest.execution_frontier = 127;

    // 1. identical digests validate cleanly on both directions.
    {
        detail::TpRetentionCoordinator coordinator(2);
        coordinator.record_lead(0, digest);
        coordinator.validate_peer(0, digest);
        coordinator.validate_lead(0, digest);
        expect(true, "identical digests must validate");
    }

    // 2. peer divergence throws with the diverging field named.
    {
        detail::TpRetentionCoordinator coordinator(2);
        coordinator.record_lead(0, digest);
        detail::RetentionDigest peer = digest;
        peer.ledger_hash += 1;
        expect(throws_with([&] { coordinator.validate_peer(0, peer); }, "ledger_hash"),
               "peer divergence must throw naming ledger_hash");
        expect(throws_with([&] { coordinator.validate_peer(0, peer); }, "rank 1 vs rank 0"),
               "divergence message must name the ranks");
    }

    // 3. lead drift against the shadow throws.
    {
        detail::TpRetentionCoordinator coordinator(2);
        coordinator.record_lead(1, digest);
        detail::RetentionDigest drifted = digest;
        drifted.execution_frontier += 1;
        expect(throws_with([&] { coordinator.validate_lead(1, drifted); }, "execution_frontier"),
               "lead drift must throw naming execution_frontier");
    }

    // 4. lanes without a recorded expectation validate trivially.
    {
        detail::TpRetentionCoordinator coordinator(1);
        coordinator.validate_peer(0, digest);
        coordinator.validate_lead(0, digest);
        expect(true, "unknown lanes must not throw");
    }

    // 5. noexcept observation stores a sticky fault rethrown by check_fault.
    {
        detail::TpRetentionCoordinator coordinator(1);
        coordinator.record_lead(0, digest);
        detail::RetentionDigest peer = digest;
        peer.text_pages_hash ^= 1;
        coordinator.observe_peer(0, peer); // must not throw
        expect(throws_with([&] { coordinator.check_fault(); }, "text_pages_hash"),
               "sticky fault must rethrow naming text_pages_hash");
        expect(throws_with([&] { coordinator.check_fault(); }, "text_pages_hash"),
               "fault must persist until consumed by policy");
    }

    // 6. invalidate_all suppresses validation until the next record_lead.
    {
        detail::TpRetentionCoordinator coordinator(1);
        coordinator.record_lead(0, digest);
        detail::RetentionDigest peer = digest;
        peer.mtp_kv_valid += 2;
        coordinator.invalidate_all();
        coordinator.validate_peer(0, peer); // must not throw
        coordinator.validate_lead(0, peer);
        coordinator.record_lead(0, peer);
        coordinator.validate_peer(0, peer);
        expect(true, "invalidated shadow must not throw");
    }

    // 7. lane bounds are enforced.
    {
        detail::TpRetentionCoordinator coordinator(1);
        expect(throws_with([&] { static_cast<void>(coordinator.record_lead(2, digest)); }, ""),
               "out-of-range lane must throw");
    }

    // 8. op journal counts.
    {
        detail::TpRetentionCoordinator coordinator(1);
        coordinator.note_op(0, "validate");
        coordinator.note_op(0, "evict");
        expect(coordinator.ops_recorded(0) == 2, "op journal must count notes");
        expect(coordinator.ops_recorded(7) == 0, "out-of-range op query must read zero");
    }

    if (failures != 0) {
        std::cerr << std::to_string(failures) << " retention coordinator checks failed\n";
        return 1;
    }
    std::cout << "tp retention coordinator ok\n";
    return 0;
}
