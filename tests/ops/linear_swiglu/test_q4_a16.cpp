#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Public numerical cases straddle each registered Q4 implementation interval. They make
        // no assertion about the private route selected for any T.
        constexpr std::array<std::int32_t, 18> kTokenCases{
            1, 2, 32, 33, 40, 41, 48, 49, 128, 129, 256, 257, 384, 385, 512, 513, 640, 641,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q4_A16",
            {QType::Q4G64_F16S, 34816, 5120, 17408, 1401U, ActivationCompute::A16}, kTokenCases);
        // Two-rank tensor-parallel N-split: gate_up [17408,5120], output 8704. The rank output
        // is the corresponding row block of the full-shape result because gate/up pairing and
        // SwiGLU are row-independent; this case is verified against the same independent oracle.
        constexpr std::array<std::int32_t, 10> kTpTokenCases{1, 2, 32, 33, 48, 49, 128, 129, 256, 641};
        const int tp_failures = run_profile(
            "LinearSwiGLU Q4_A16 TP",
            {QType::Q4G64_F16S, 17408, 5120, 8704, 1423U, ActivationCompute::A16}, kTpTokenCases);
        std::cout << ((failures + tp_failures) == 0 ? "OK" : "FAIL")
                  << " LinearSwiGLU Q4_A16 correctness\n";
        return (failures + tp_failures) == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
