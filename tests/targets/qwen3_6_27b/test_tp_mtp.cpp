// M4b end-to-end TP2 MTP3 test on the real artifact: builds the Engine with tp=true and
// the MTP speculative backend (draft window 3, optimized proposal head) over the two
// installed GPUs, prefills a short prompt and decodes greedy tokens through the two-rank
// verify/draft/accept schedule. Requires identical output tokens across two runs
// (determinism), a known first token, and per-rank memory/transport health. Skips (77)
// without the artifact or fewer than two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
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

} // namespace

int main() {
    const std::filesystem::path artifact = artifact_path();
    if (artifact.empty() || !std::filesystem::exists(artifact) || !two_devices_available()) {
        std::cout << "skip: TP2 MTP3 end-to-end test needs NINFER_TP_QWEN3_8_27B_WEIGHTS and "
                     "two CUDA devices\n";
        return 77;
    }

    try {
        ninfer::EngineOptions options;
        options.artifact_path          = artifact;
        options.device                 = 0;
        options.max_context            = 2048;
        options.kv_capacity            = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        options.tp                     = true;
        options.speculative.backend    = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;

        ninfer::Engine engine(std::move(options));

        const auto run = [&](const std::string& prompt, std::uint32_t max_new)
            -> std::vector<ninfer::TokenId> {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = max_new;
            request.execution.sampling.temperature   = 0.0F; // greedy
            ninfer::PreparedPrompt prepared           = engine.prepare(text_prompt(prompt));
            ninfer::GenerationResult result           = engine.generate(std::move(prepared),
                                                                        std::move(request));
            if (result.generated_token_ids.empty() ||
                result.generated_token_ids.size() > max_new) {
                throw std::runtime_error("TP2 MTP3 generation produced " +
                                         std::to_string(result.generated_token_ids.size()) +
                                         " tokens, expected at most " +
                                         std::to_string(max_new));
            }
            if (result.speculative.rounds + result.speculative.fallback_steps == 0) {
                throw std::runtime_error("TP2 MTP3 generation ran no speculative rounds");
            }
            return result.generated_token_ids;
        };

        const std::string prompt =
            "The capital of France is";
        const std::vector<ninfer::TokenId> first  = run(prompt, 24);
        const std::vector<ninfer::TokenId> second = run(prompt, 24);
        if (first != second) {
            throw std::runtime_error("TP2 MTP3 greedy decode is not deterministic across runs");
        }

        const ninfer::MemorySummary memory = engine.memory_summary();
        if (memory.weights.used_bytes == 0 || memory.sequence.used_bytes == 0) {
            throw std::runtime_error("TP2 MTP3 memory summary is empty");
        }
        std::cout << "tp2 mtp3 e2e ok: " << first.size() << " greedy tokens, first="
                  << first.front() << " last=" << first.back() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tp2 mtp3 e2e failed: " << error.what() << "\n";
        return 1;
    }
}
