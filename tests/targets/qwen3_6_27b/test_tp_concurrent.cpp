// TP2 multi-concurrency end-to-end test on the real artifact: proves that the two-rank
// tensor-parallel engine serves concurrent requests with the same greedy results as the
// single-lane engine.
//
// Phase C1: an engine with max_concurrency=1 runs a fixed case set sequentially; the France
// case must reproduce the documented spot gate (README "TP2 measured results").
// Phase C8: a fresh engine with max_concurrency=8 runs the same case set submitted
// CONCURRENTLY, one std::async per case; every case's token ids must equal its C1 baseline
// exactly. A sequential re-run on the warm C8 engine must still match the baseline.
// Reuse case: with the TP prefix-reuse opt-in on, an exact immediate repeat of a prompt must
// return the identical tokens (exercises the retained-tail sampling path).
// Phase AR (abort-resume): on a small-chunk engine, a request cancelled mid-prefill must
// park its completed-chunk watermark as a retained prefix; the identical retry must resume
// from that watermark (AppendAtFrontier, not FullReset, strictly less prompt computed) and
// produce tokens identical to a cold full prefill of the same prompt.
//
// Skips (77) without NINFER_TP_QWEN3_8_27B_WEIGHTS or fewer than two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_TP_QWEN3_8_27B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return {};
}

bool two_devices_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count >= 2;
}

ninfer::PromptInput text_prompt(const std::string& text) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

struct Case {
    std::string   name;
    std::string   prompt;
    std::uint32_t max_new;
};

const std::vector<Case> kCases = {
    {"france", "The capital of France is", 16},
    {"colors", "Name three primary colors.", 24},
    {"rain", "Write one short sentence about rain.", 24},
    {"math", "What is 17 plus 25?", 24},
    {"river", "Give one fact about the Amazon river.", 24},
    {"long", "Explain in a few sentences why the sky is blue.", 64},
};

// README "TP2 measured results" spot gate for the France prompt (token-identical with and
// without MTP3).
const std::vector<ninfer::TokenId> kFranceSpotGate = {760,  6511,   314,    9338,   369,
                                                      2972, 57590, 159034, 248046};

ninfer::RequestOptions greedy_request(std::uint32_t max_new, bool tp_reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = max_new;
    options.execution.sampling.temperature    = 0.0F; // TP requires greedy
    options.execution.allow_prefix_reuse      = tp_reuse;
    options.execution.tp_prefix_reuse         = tp_reuse;
    return options;
}

ninfer::EngineOptions tp_options(const std::filesystem::path& artifact,
                                 std::uint32_t max_concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.device                    = 0;
    // The KV pool rule kv ≤ max_concurrency × max_context binds the C1 phase hardest, so
    // keep kv == max_context; six short lanes still fit the pool easily.
    options.max_context               = 1024;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.max_concurrency           = max_concurrency;
    options.tp                        = true;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

std::vector<ninfer::TokenId> generate(ninfer::Engine& engine, const Case& test_case) {
    ninfer::GenerationResult result =
        engine.generate(engine.prepare(text_prompt(test_case.prompt)),
                        greedy_request(test_case.max_new, false));
    if (result.generated_token_ids.empty() ||
        result.generated_token_ids.size() > test_case.max_new) {
        throw std::runtime_error("case " + test_case.name + " produced " +
                                 std::to_string(result.generated_token_ids.size()) +
                                 " tokens, expected at most " + std::to_string(test_case.max_new));
    }
    return result.generated_token_ids;
}

void expect_equal(const Case& test_case, const std::vector<ninfer::TokenId>& actual,
                  const std::vector<ninfer::TokenId>& baseline, const char* phase) {
    if (actual != baseline) {
        std::string actual_text;
        for (const ninfer::TokenId id : actual) { actual_text += std::to_string(id) + " "; }
        std::string baseline_text;
        for (const ninfer::TokenId id : baseline) { baseline_text += std::to_string(id) + " "; }
        throw std::runtime_error(std::string(phase) + ": case " + test_case.name +
                                 " diverged from the C1 baseline\n  baseline: " + baseline_text +
                                 "\n  actual:   " + actual_text);
    }
}

// Phase AR: cancel a multi-chunk prefill after its first completed chunk, then prove the
// identical retry resumes from the retained watermark instead of re-prefilling from zero —
// the coding-agent timeout/retry pattern from the serving log.
int exercise_abort_resume(const std::filesystem::path& artifact) {
    ninfer::EngineOptions options = tp_options(artifact, 2);
    options.prefill_chunk         = 128;
    ninfer::Engine engine(options);

    // Two prompts sharing no token prefix, so each request's lane plan reuses nothing from
    // the other's retained state.
    const auto make_ids = [](ninfer::TokenId first) {
        std::vector<ninfer::TokenId> ids;
        ids.reserve(640);
        for (std::uint32_t i = 0; i < 640; ++i) {
            ids.push_back(static_cast<ninfer::TokenId>(first + i % 900));
        }
        return ids;
    };
    const std::vector<ninfer::TokenId> other_ids = make_ids(100);
    const std::vector<ninfer::TokenId> abort_ids = make_ids(50000);

    // An unrelated completed request parks a retained lane the abort prompt cannot match.
    (void)engine.generate(engine.prepare_tokens(other_ids), greedy_request(4, false));

    const std::uint64_t prefill_before = engine.runtime_stats().computed_prefill_tokens;
    std::atomic<bool> cancel_requested{false};
    ninfer::GenerationHandle handle =
        engine.submit(engine.prepare_tokens(abort_ids), greedy_request(256, true));
    // Let at least one 128-token chunk land, then cancel; the request must still be
    // mid-prefill (five chunks total on this engine).
    for (int spin = 0; spin < 1500; ++spin) {
        if (engine.runtime_stats().computed_prefill_tokens - prefill_before >= 128) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    cancel_requested.store(true);
    ninfer::GenerationResult cancelled = handle.wait(
        nullptr, ninfer::CancellationView([&] { return cancel_requested.load(); }));
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        throw std::runtime_error("abort-resume request did not cancel");
    }

    const std::uint64_t resumed_prefill_before = engine.runtime_stats().computed_prefill_tokens;
    ninfer::GenerationResult resumed =
        engine.generate(engine.prepare_tokens(abort_ids), greedy_request(8, true));
    const std::uint64_t resumed_computed =
        engine.runtime_stats().computed_prefill_tokens - resumed_prefill_before;
    if (resumed.reused_prompt_tokens < 128 || resumed.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        throw std::runtime_error("abort-resume retry did not resume from the retained watermark: reused=" +
                                 std::to_string(resumed.reused_prompt_tokens));
    }
    if (resumed_computed >= 640) {
        throw std::runtime_error("abort-resume retry re-prefilled the whole prompt (computed=" +
                                 std::to_string(resumed_computed) + ")");
    }

    // Cold full prefill of the same prompt must be token-identical to the resumed decode.
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(abort_ids), greedy_request(8, false));
    if (cold.reused_prompt_tokens != 0) {
        throw std::runtime_error("cold baseline unexpectedly reused a prefix: reused=" +
                                 std::to_string(cold.reused_prompt_tokens));
    }
    if (resumed.generated_token_ids != cold.generated_token_ids) {
        std::string resumed_text;
        for (const ninfer::TokenId id : resumed.generated_token_ids) {
            resumed_text += std::to_string(id) + " ";
        }
        std::string cold_text;
        for (const ninfer::TokenId id : cold.generated_token_ids) { cold_text += std::to_string(id) + " "; }
        throw std::runtime_error("abort-resume decode diverged from the cold baseline\n  resumed: " +
                                 resumed_text + "\n  cold:    " + cold_text);
    }

    // Mid-decode cancel: the cancelled round's committed watermark keeps the lane coherent,
    // but a raw re-send of the bare prompt under MTP cannot append into the longer resident
    // sequence (the bridge would consume a tail hidden past the prompt boundary and the GDN
    // slot sits at the resident frontier), so the retry legitimately full-resets. What must
    // hold is correctness: the retry decodes token-identically to the cold baseline.
    {
        cancel_requested.store(false);
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            cancel_requested.store(true);
        });
        ninfer::GenerationResult generating = engine.generate(
            engine.prepare_tokens(abort_ids), greedy_request(256, true), nullptr,
            ninfer::CancellationView([&] { return cancel_requested.load(); }));
        canceller.join();
        if (generating.finish_reason != ninfer::FinishReason::Cancelled) {
            throw std::runtime_error("mid-decode request did not cancel: reason=" +
                                     std::to_string(static_cast<int>(generating.finish_reason)));
        }
        const ninfer::GenerationResult after_decode_cancel =
            engine.generate(engine.prepare_tokens(abort_ids), greedy_request(8, true));
        if (after_decode_cancel.generated_token_ids != cold.generated_token_ids) {
            throw std::runtime_error(
                "mid-decode cancel retry diverged from the cold baseline");
        }
    }
    return 0;
}

} // namespace

int main() {
    const std::filesystem::path artifact = artifact_path();
    if (artifact.empty() || !std::filesystem::exists(artifact) || !two_devices_available()) {
        std::cout << "skip: TP2 concurrency test needs NINFER_TP_QWEN3_8_27B_WEIGHTS and two "
                     "CUDA devices\n";
        return 77;
    }

    try {
        std::vector<std::vector<ninfer::TokenId>> baseline;
        {
            ninfer::Engine engine(tp_options(artifact, 1));
            for (const Case& test_case : kCases) {
                baseline.push_back(generate(engine, test_case));
            }

            const std::vector<ninfer::TokenId>& france = baseline.front();
            if (france.size() < kFranceSpotGate.size() ||
                !std::equal(kFranceSpotGate.begin(), kFranceSpotGate.end(), france.begin())) {
                std::string actual_text;
                for (const ninfer::TokenId id : france) {
                    actual_text += std::to_string(id) + " ";
                }
                throw std::runtime_error("TP2 C1 France spot gate mismatch: " + actual_text);
            }

            // Exact immediate repeat with the TP reuse opt-in: the retained-tail sampling path
            // must return the identical tokens (never the full-vocab sample throw).
            const std::vector<ninfer::TokenId> repeat =
                [&] {
                    ninfer::GenerationResult result = engine.generate(
                        engine.prepare(text_prompt(kCases.front().prompt)),
                        greedy_request(kCases.front().max_new, true));
                    return result.generated_token_ids;
                }();
            expect_equal(kCases.front(), repeat, france, "tp-reuse repeat");
        }

        {
            ninfer::Engine engine(tp_options(artifact, 8));
            std::vector<std::future<std::vector<ninfer::TokenId>>> futures;
            futures.reserve(kCases.size());
            for (const Case& test_case : kCases) {
                futures.push_back(
                    std::async(std::launch::async, [&engine, &test_case] {
                        return generate(engine, test_case);
                    }));
            }
            for (std::size_t index = 0; index < kCases.size(); ++index) {
                expect_equal(kCases[index], futures[index].get(), baseline[index],
                             "tp c8 concurrent");
            }

            // Warm sequential re-run on the C8 engine: lane reuse across unrelated prompts
            // must not corrupt results.
            for (std::size_t index = 0; index < kCases.size(); ++index) {
                expect_equal(kCases[index], generate(engine, kCases[index]), baseline[index],
                             "tp c8 sequential");
            }
        }

        if (const int result = exercise_abort_resume(artifact); result != 0) { return result; }

        std::cout << "tp2 concurrency ok: " << kCases.size()
                  << " cases identical across C1/C8/concurrent/warm-sequential phases"
                  << " (+abort-resume)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tp2 concurrency failed: " << error.what() << "\n";
        return 1;
    }
}
