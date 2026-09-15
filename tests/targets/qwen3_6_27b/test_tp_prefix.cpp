// TP2 compatible-prefix reuse test on the real artifact: builds the Engine with tp=true and
// MTP3 over the two installed GPUs, then exercises the retained-sequence reuse paths with the
// TP opt-in gate (execution.tp_prefix_reuse) that TpProgram::plan_request_base honours.
//
// Scenario A (raw tokens, exact append): a continuation prompt that extends the retained
// ledger must report append_frontier reuse, and its greedy tokens must be identical to a
// cold full-reset run of the same prompt. MTP must keep proposing across the reuse boundary.
//
// Scenario B (chat template, agent turn shape): a follow-up turn that re-sends the full
// history (user + assistant reply + new user) must reuse the shared prefix rather than
// full-reset, and again match the cold full-reset run token-for-token.
//
// Skips (77) without NINFER_TP_QWEN3_8_27B_WEIGHTS or fewer than two CUDA devices.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
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

ninfer::RequestOptions greedy_request(std::uint32_t max_new, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = max_new;
    options.execution.sampling.temperature    = 0.0F; // TP requires greedy
    options.execution.allow_prefix_reuse      = reuse;
    options.execution.tp_prefix_reuse         = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::PromptInput chat_turn(const std::string& user) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = user, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::PromptInput chat_follow_up(const std::string& first_user, const std::string& assistant,
                                   const std::string& second_user) {
    ninfer::PromptInput input;
    ninfer::ChatMessage u1;
    u1.role = "user";
    u1.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = first_user, .media = {}});
    ninfer::ChatMessage a1;
    a1.role = "assistant";
    a1.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = assistant, .media = {}});
    ninfer::ChatMessage u2;
    u2.role = "user";
    u2.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = second_user, .media = {}});
    input.messages.push_back(std::move(u1));
    input.messages.push_back(std::move(a1));
    input.messages.push_back(std::move(u2));
    input.options.enable_thinking = false;
    return input;
}

const ninfer::TokenId kStopFriendlyNewline = 198;

} // namespace

int exercise_raw_token_append(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_request(5, false));
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "append: first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(kStopFriendlyNewline);

    // Cold identity baseline for the same continuation prompt, reuse disabled (full reset).
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_request(5, false));

    // Re-establish the retained sequence, then run the same continuation with reuse on.
    const ninfer::GenerationResult again =
        engine.generate(engine.prepare_tokens(prompt), greedy_request(5, true));
    if (again.generated_token_ids != first.generated_token_ids) {
        std::cerr << "append: greedy full reset is not deterministic across runs\n";
        return 1;
    }

    const ninfer::GenerationResult warm =
        engine.generate(engine.prepare_tokens(continuation), greedy_request(5, true));

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (warm.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append: reuse count is " << warm.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }
    if (warm.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier) {
        std::cerr << "append: reuse path is not append_frontier\n";
        return 1;
    }
    if (warm.generated_token_ids != cold.generated_token_ids) {
        std::cerr << "append: warm reused decode diverges from the cold full-reset baseline\n";
        return 1;
    }
    if (warm.speculative.rounds == 0) {
        std::cerr << "append: MTP stopped proposing across the reuse boundary\n";
        return 1;
    }
    return 0;
}

int exercise_chat_turn(ninfer::Engine& engine) {
    const std::string first_user = "What is the capital of France? Answer with just the city.";

    const ninfer::GenerationResult turn1 =
        engine.generate(engine.prepare(chat_turn(first_user)), greedy_request(16, true));
    if (turn1.content.empty()) {
        std::cerr << "chat: first turn produced no content\n";
        return 1;
    }

    const std::string second_user = "And its population? One number.";
    const ninfer::PromptInput follow_up =
        chat_follow_up(first_user, turn1.content, second_user);

    const ninfer::GenerationResult warm =
        engine.generate(engine.prepare(follow_up), greedy_request(16, true));
    if (warm.reused_prompt_tokens == 0) {
        std::cerr << "chat: follow-up turn full-reset instead of reusing the shared prefix\n";
        return 1;
    }
    if (warm.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << "chat: follow-up reuse path is full_reset\n";
        return 1;
    }

    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare(chat_follow_up(first_user, turn1.content, second_user)),
                        greedy_request(16, false));
    if (warm.generated_token_ids != cold.generated_token_ids) {
        std::cerr << "chat: warm reused turn diverges from the cold full-reset baseline\n";
        return 1;
    }
    return 0;
}

int main() {
    const std::filesystem::path artifact = artifact_path();
    if (artifact.empty() || !std::filesystem::exists(artifact) || !two_devices_available()) {
        std::cout << "skip: TP2 prefix reuse test needs NINFER_TP_QWEN3_8_27B_WEIGHTS and two "
                     "CUDA devices\n";
        return 77;
    }

    try {
        ninfer::EngineOptions options;
        options.artifact_path             = artifact;
        options.device                    = 0;
        options.max_context               = 2048;
        options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        options.tp                        = true;
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;

        ninfer::Engine engine(std::move(options));

        if (const int result = exercise_raw_token_append(engine); result != 0) { return result; }
        if (const int result = exercise_chat_turn(engine); result != 0) { return result; }

        std::cout << "tp2 prefix reuse ok: append_frontier + chat turn reuse, warm==cold "
                     "identity, MTP bridged\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tp2 prefix reuse failed: " << error.what() << "\n";
        return 1;
    }
}
