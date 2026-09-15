// Two-GPU correctness test for core/tp_link.h on real devices: exact allreduce_add,
// vocab-half argmax with a cross-rank tie, 128 chained dependent allreduces, reset()
// reuse, and allgather. Skips (77) when fewer than two CUDA devices exist.

#include "core/tp_link.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void ck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(err) << '\n';
        ++failures;
    }
}

// bf16 <-> fp32 bit helpers (round-to-nearest-even), mirroring tests/ops/op_tester.h.
inline float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0f;
    std::memcpy(&value, &wide, sizeof value);
    return value;
}

inline std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return static_cast<std::uint16_t>((bits >> 16) | 0x0040u); }
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

// Small device buffer bound to one rank's device.
class RankBuffer {
public:
    RankBuffer(int device, std::size_t bytes) : device_(device) {
        ck(cudaSetDevice(device), "RankBuffer cudaSetDevice");
        ck(cudaMalloc(&data_, bytes), "RankBuffer cudaMalloc");
    }

    ~RankBuffer() {
        if (data_ != nullptr) {
            ck(cudaSetDevice(device_), "RankBuffer cudaSetDevice");
            ck(cudaFree(data_), "RankBuffer cudaFree");
        }
    }

    RankBuffer(const RankBuffer&)            = delete;
    RankBuffer& operator=(const RankBuffer&) = delete;

    [[nodiscard]] __nv_bfloat16* as_bf16() { return static_cast<__nv_bfloat16*>(data_); }
    [[nodiscard]] std::int32_t* as_i32() { return static_cast<std::int32_t*>(data_); }

    void upload(const void* host, std::size_t bytes, cudaStream_t stream) {
        ck(cudaSetDevice(device_), "RankBuffer cudaSetDevice");
        ck(cudaMemcpyAsync(data_, host, bytes, cudaMemcpyHostToDevice, stream), "upload");
        ck(cudaStreamSynchronize(stream), "upload sync");
    }

    void download(void* host, std::size_t bytes, cudaStream_t stream) {
        ck(cudaSetDevice(device_), "RankBuffer cudaSetDevice");
        ck(cudaMemcpyAsync(host, data_, bytes, cudaMemcpyDeviceToHost, stream), "download");
        ck(cudaStreamSynchronize(stream), "download sync");
    }

private:
    int device_      = 0;
    void* data_      = nullptr;
};

bool report(const char* label, bool ok) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << '\n';
    if (!ok) { ++failures; }
    return ok;
}

bool healthy(ninfer::TpLink& link, const char* label) {
    const ninfer::TpLinkHealth health = link.check_health();
    if (!health.ok()) {
        std::cerr << label << ": bounded-spin timeout, error words " << health.error[0] << '/'
                  << health.error[1] << '\n';
        return false;
    }
    return true;
}

// Exact elementwise allreduce: every input and the fp32-accumulated sum land on
// exactly representable bf16 values, so both ranks must match the host oracle bitwise.
bool run_allreduce_add(ninfer::TpLink& link, cudaStream_t* streams, std::size_t n) {
    std::vector<std::uint16_t> host_a(n), host_b(n), expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = static_cast<float>(i % 7) - 3.0f;
        const float b = static_cast<float>(i % 11) * 0.5f - 2.0f;
        host_a[i]     = f32_to_bf16(a);
        host_b[i]     = f32_to_bf16(b);
        expected[i]   = f32_to_bf16(bf16_to_f32(host_a[i]) + bf16_to_f32(host_b[i]));
    }

    RankBuffer tensor_a(0, n * sizeof(std::uint16_t));
    RankBuffer tensor_b(1, n * sizeof(std::uint16_t));
    tensor_a.upload(host_a.data(), host_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.upload(host_b.data(), host_b.size() * sizeof(std::uint16_t), streams[1]);

    link.allreduce_add(streams[0], streams[1], tensor_a.as_bf16(), tensor_b.as_bf16(), n);

    std::vector<std::uint16_t> out_a(n), out_b(n);
    tensor_a.download(out_a.data(), out_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.download(out_b.data(), out_b.size() * sizeof(std::uint16_t), streams[1]);

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (out_a[i] != expected[i] || out_b[i] != expected[i]) {
            if (mismatches < 4) {
                std::cerr << "  n=" << n << " i=" << i << " got " << out_a[i] << '/' << out_b[i]
                          << " expected " << expected[i] << '\n';
            }
            ++mismatches;
        }
    }
    const bool ok = mismatches == 0 && healthy(link, "allreduce_add");
    const std::string label = "tp_link allreduce_add n=" + std::to_string(n);
    return report(label.c_str(), ok);
}

// Vocab halves (n_a != n_b); tie_case puts the same peak value on both halves so the
// lower global index must win, otherwise the peak lives only on rank b.
bool run_allreduce_argmax(ninfer::TpLink& link, cudaStream_t* streams, bool tie_case) {
    const std::size_t n_a = 8192;
    const std::size_t n_b = 6144;
    const std::int32_t base_b = 8192;
    std::vector<std::uint16_t> host_a(n_a), host_b(n_b);
    for (std::size_t i = 0; i < n_a; ++i) {
        host_a[i] = f32_to_bf16(-24.0f + static_cast<float>((i * 1664525u) % 2048u) * (1.0f / 256.0f));
    }
    for (std::size_t i = 0; i < n_b; ++i) {
        host_b[i] = f32_to_bf16(-24.0f + static_cast<float>((i * 22695477u) % 2048u) * (1.0f / 256.0f));
    }
    std::int32_t expected = 0;
    if (tie_case) {
        host_a[100] = f32_to_bf16(32.0f);
        host_b[3]   = f32_to_bf16(32.0f);
        expected    = 100; // equal values: lower global index wins
    } else {
        host_b[17] = f32_to_bf16(32.0f);
        expected   = base_b + 17;
    }

    RankBuffer logits_a(0, n_a * sizeof(std::uint16_t));
    RankBuffer logits_b(1, n_b * sizeof(std::uint16_t));
    RankBuffer token_a(0, sizeof(std::int32_t));
    RankBuffer token_b(1, sizeof(std::int32_t));
    logits_a.upload(host_a.data(), host_a.size() * sizeof(std::uint16_t), streams[0]);
    logits_b.upload(host_b.data(), host_b.size() * sizeof(std::uint16_t), streams[1]);
    const std::int32_t guard = -1;
    token_a.upload(&guard, sizeof guard, streams[0]);
    token_b.upload(&guard, sizeof guard, streams[1]);

    link.allreduce_argmax(streams[0], streams[1], logits_a.as_bf16(), n_a, 0, logits_b.as_bf16(),
                          n_b, base_b, token_a.as_i32(), token_b.as_i32());

    std::int32_t out_a = -1, out_b = -1;
    token_a.download(&out_a, sizeof out_a, streams[0]);
    token_b.download(&out_b, sizeof out_b, streams[1]);
    const bool ok = out_a == expected && out_b == expected && healthy(link, "allreduce_argmax");
    if (!ok) { std::cerr << "  argmax tie=" << tie_case << " got " << out_a << '/' << out_b << " expected " << expected << '\n'; }
    return report(tie_case ? "tp_link allreduce_argmax cross-rank tie -> lower index"
                           : "tp_link allreduce_argmax b-half winner",
                  ok);
}

// 128 dependent allreduces over the same tensors (seeds v, 2v with v = 2^-120): each
// result doubles exactly, ending at 3 * 2^7 = 384 with zero rounding anywhere.
bool run_chained_allreduce(ninfer::TpLink& link, cudaStream_t* streams) {
    const std::size_t n = 5120;
    const float seed    = std::ldexp(1.0f, -120);
    const std::vector<std::uint16_t> host_a(n, f32_to_bf16(seed));
    const std::vector<std::uint16_t> host_b(n, f32_to_bf16(2.0f * seed));

    RankBuffer tensor_a(0, n * sizeof(std::uint16_t));
    RankBuffer tensor_b(1, n * sizeof(std::uint16_t));
    tensor_a.upload(host_a.data(), host_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.upload(host_b.data(), host_b.size() * sizeof(std::uint16_t), streams[1]);

    constexpr int kRounds = 128;
    const auto start = std::chrono::steady_clock::now();
    for (int round = 0; round < kRounds; ++round) {
        link.allreduce_add(streams[0], streams[1], tensor_a.as_bf16(), tensor_b.as_bf16(), n);
    }
    ck(cudaSetDevice(0), "chain sync");
    ck(cudaStreamSynchronize(streams[0]), "chain sync");
    ck(cudaSetDevice(1), "chain sync");
    ck(cudaStreamSynchronize(streams[1]), "chain sync");
    const auto stop = std::chrono::steady_clock::now();
    const double us_per_op =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(stop - start).count() /
        kRounds;

    std::vector<std::uint16_t> out_a(n), out_b(n);
    tensor_a.download(out_a.data(), out_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.download(out_b.data(), out_b.size() * sizeof(std::uint16_t), streams[1]);
    const std::uint16_t expected = f32_to_bf16(384.0f);
    std::size_t mismatches       = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (out_a[i] != expected || out_b[i] != expected) { ++mismatches; }
    }
    const bool ok = mismatches == 0 && healthy(link, "chained allreduce");
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << "tp_link chained allreduce x" << kRounds
              << " n=" << n << ": " << us_per_op << " us/op\n";
    if (!ok) { ++failures; }
    return ok;
}

// reset() then a fresh exchange with different data must stay exact.
bool run_reset_reuse(ninfer::TpLink& link, cudaStream_t* streams) {
    link.reset(streams[0], streams[1]);

    const std::size_t n = 5120;
    std::vector<std::uint16_t> host_a(n), host_b(n), expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = static_cast<float>(i % 5) - 2.0f;
        const float b = static_cast<float>(i % 13) * 0.25f - 1.0f;
        host_a[i]     = f32_to_bf16(a);
        host_b[i]     = f32_to_bf16(b);
        expected[i]   = f32_to_bf16(bf16_to_f32(host_a[i]) + bf16_to_f32(host_b[i]));
    }

    RankBuffer tensor_a(0, n * sizeof(std::uint16_t));
    RankBuffer tensor_b(1, n * sizeof(std::uint16_t));
    tensor_a.upload(host_a.data(), host_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.upload(host_b.data(), host_b.size() * sizeof(std::uint16_t), streams[1]);
    link.allreduce_add(streams[0], streams[1], tensor_a.as_bf16(), tensor_b.as_bf16(), n);

    std::vector<std::uint16_t> out_a(n), out_b(n);
    tensor_a.download(out_a.data(), out_a.size() * sizeof(std::uint16_t), streams[0]);
    tensor_b.download(out_b.data(), out_b.size() * sizeof(std::uint16_t), streams[1]);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (out_a[i] != expected[i] || out_b[i] != expected[i]) { ++mismatches; }
    }
    const bool ok = mismatches == 0 && healthy(link, "reset reuse");
    return report("tp_link reset + reuse allreduce_add n=5120", ok);
}

// dst = concat(a, b) on both ranks; odd sizes exercise the element tail.
bool run_allgather(ninfer::TpLink& link, cudaStream_t* streams) {
    const std::size_t n_a = 3000;
    const std::size_t n_b = 2001;
    std::vector<std::uint16_t> host_a(n_a), host_b(n_b), expected(n_a + n_b);
    for (std::size_t i = 0; i < n_a; ++i) {
        host_a[i]       = f32_to_bf16(static_cast<float>(i % 97));
        expected[i]     = host_a[i];
    }
    for (std::size_t i = 0; i < n_b; ++i) {
        host_b[i]           = f32_to_bf16(-static_cast<float>(i % 89) - 0.5f);
        expected[n_a + i]   = host_b[i];
    }

    const std::size_t total = (n_a + n_b) * sizeof(std::uint16_t);
    RankBuffer src_a(0, n_a * sizeof(std::uint16_t));
    RankBuffer src_b(1, n_b * sizeof(std::uint16_t));
    RankBuffer dst_a(0, total);
    RankBuffer dst_b(1, total);
    src_a.upload(host_a.data(), host_a.size() * sizeof(std::uint16_t), streams[0]);
    src_b.upload(host_b.data(), host_b.size() * sizeof(std::uint16_t), streams[1]);
    const std::uint16_t guard = 0xcccc;
    for (int side = 0; side < 2; ++side) {
        std::vector<std::uint16_t> guards(n_a + n_b, guard);
        (side == 0 ? dst_a : dst_b).upload(guards.data(), total, streams[side]);
    }

    link.allgather(streams[0], streams[1], src_a.as_bf16(), src_b.as_bf16(), dst_a.as_bf16(),
                   dst_b.as_bf16(), n_a, n_b);

    std::vector<std::uint16_t> out_a(n_a + n_b), out_b(n_a + n_b);
    dst_a.download(out_a.data(), total, streams[0]);
    dst_b.download(out_b.data(), total, streams[1]);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < n_a + n_b; ++i) {
        if (out_a[i] != expected[i] || out_b[i] != expected[i]) {
            if (mismatches < 8) {
                std::cerr << "  i=" << i << " got " << out_a[i] << '/' << out_b[i] << " expected "
                          << expected[i] << (out_a[i] == guard || out_b[i] == guard ? " (guard)" : "")
                          << '\n';
            }
            ++mismatches;
        }
    }
    const bool ok = mismatches == 0 && healthy(link, "allgather");
    return report("tp_link allgather 3000+2001 on both ranks", ok);
}

} // namespace

int main() {
    int count             = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (count < 2) {
        std::cout << "SKIP: needs two CUDA devices, found " << count << '\n';
        return 77;
    }

    cudaStream_t streams[2] = {};
    for (int side = 0; side < 2; ++side) {
        ck(cudaSetDevice(side), "test cudaSetDevice");
        ck(cudaStreamCreateWithFlags(&streams[side], cudaStreamNonBlocking),
           "test cudaStreamCreateWithFlags");
    }
    if (failures != 0) { return 1; }

    try {
        constexpr std::size_t kMaxElements = 20480;
        // Both transport modes stay covered: Spin is the engine's product path,
        // StreamWait remains selectable through the TpLink mode param.
        for (const ninfer::TpLinkMode mode :
             {ninfer::TpLinkMode::StreamWait, ninfer::TpLinkMode::Spin}) {
            std::cout << "tp_link mode: " << (mode == ninfer::TpLinkMode::StreamWait
                                                  ? "stream-wait"
                                                  : "spin")
                      << '\n';
            ninfer::TpLink link(0, 1, kMaxElements, mode);

            run_allreduce_add(link, streams, 5120);
            run_allreduce_add(link, streams, 20480);
            run_allreduce_argmax(link, streams, false);
            run_allreduce_argmax(link, streams, true);
            run_chained_allreduce(link, streams);
            run_reset_reuse(link, streams);
            run_allgather(link, streams);
        }
    } catch (const std::exception& error) {
        std::cerr << "tp_link test exception: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0) {
        std::cerr << "tp_link test failed with " << failures << " failure(s)\n";
        return 1;
    }
    return 0;
}
