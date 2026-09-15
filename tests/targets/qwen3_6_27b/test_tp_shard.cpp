// M3a rank-sharded materialization test on the real artifact: rank 0 loads on dev0 and rank 1 on
// dev1, each materializing only its shard of the Text/MTP weights into its own DeviceArena.
// Verifies the single-GPU plan is unchanged (whole-tensor placements only), per-rank arena bytes
// land near half of the single-GPU weight plan, both ranks load with their own device validation,
// and D2H byte extracts match host-computed expectations from the artifact payload. Skips (77)
// without the artifact or fewer than two CUDA devices.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/load/tp_shard.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_6_27b::detail;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::Reader;
using Package = ninfer::targets::qwen3_6_27b::Package;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::targets::qwen3_6::StartupFeatures text_features() {
    return {
        .vision        = false,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_TP_QWEN3_8_27B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    if (const char* value = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path("models/qwen3_8_27b.ninfer");
}

int failures = 0;

void check(bool ok, const std::string& label) {
    if (!ok) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

struct RankLoad {
    std::unique_ptr<ninfer::DeviceContext> device;
    ninfer::artifact::MaterializedArtifact materialized;
    std::uint64_t arena_bytes = 0;
};

RankLoad load_rank(const Reader& reader, WeightsProfile profile, std::uint8_t rank,
                   int device_id) {
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, text_features(), rank, true);
    RankLoad out;
    out.device       = std::make_unique<ninfer::DeviceContext>(device_id);
    out.materialized = ninfer::artifact::materialize(reader, plan.materialization, *out.device);
    out.arena_bytes  = plan.materialization.device_capacity_bytes;
    CUDA_CHECK(cudaSetDevice(device_id));
    return out;
}

// row-split-k128-v1 plane geometry of one stored tensor.
struct RowSplitGeometryLite {
    std::uint64_t rows        = 0;
    std::uint64_t groups      = 0; // per row, over padded K
    std::uint64_t low_group   = 0;
    std::uint64_t high_group  = 0;
    const std::byte* payload  = nullptr;
    std::uint64_t low_offset  = 0;
    std::uint64_t high_offset = 0;
    std::uint64_t scale_offset = 0;
};

RowSplitGeometryLite geometry_of(const Reader& reader, const char* name, NumericFormat format) {
    const auto* object = reader.find(name);
    require(object != nullptr, "spot-check tensor is missing from the artifact");
    const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(object);
    require(tensor != nullptr && tensor->shape.size() == 2, "spot-check tensor is not rank two");
    const auto payload = reader.payload(*object);

    RowSplitGeometryLite out;
    out.rows        = tensor->shape[0];
    out.groups      = (tensor->shape[1] + 127) / 128 * 128 / (format == NumericFormat::W8G32_F16S ? 32 : 64);
    out.low_group   = 32;
    out.high_group  = format == NumericFormat::Q5G64_F16S   ? 8
                      : format == NumericFormat::Q6G64_F16S ? 16
                                                            : 0;
    out.payload     = payload.data.data();
    const std::uint64_t total_groups = out.rows * out.groups;
    std::uint64_t offset             = total_groups * 32;
    out.low_offset                   = 0;
    out.high_offset                  = (offset + 255) / 256 * 256;
    offset                           = out.high_offset + (total_groups * out.high_group + 255) / 256 * 256;
    out.scale_offset                 = offset;
    return out;
}

std::size_t object_index(const Reader& reader, const char* name) {
    const auto& objects = reader.objects();
    const auto* object  = reader.find(name);
    require(object != nullptr, "spot-check tensor is missing from the artifact");
    return static_cast<std::size_t>(object - objects.data());
}

// Host-extracted expected bytes for `count` consecutive groups of one source row.
struct PlaneBytes {
    std::vector<std::byte> codes;
    std::vector<std::byte> high;
    std::vector<std::byte> scales;
};

PlaneBytes host_row_groups(const Reader& reader, const char* name, NumericFormat format,
                           std::uint64_t row, std::uint64_t first_group, std::uint64_t count) {
    const RowSplitGeometryLite geometry = geometry_of(reader, name, format);
    require(row < geometry.rows && first_group + count <= geometry.groups,
            "spot-check row/group is outside the stored tensor");
    PlaneBytes out;
    out.codes.resize(count * geometry.low_group);
    out.scales.resize(count * 2);
    if (geometry.high_group > 0) { out.high.resize(count * geometry.high_group); }
    for (std::uint64_t g = 0; g < count; ++g) {
        const std::uint64_t source_group = row * geometry.groups + first_group + g;
        std::memcpy(out.codes.data() + g * geometry.low_group,
                    geometry.payload + geometry.low_offset + source_group * geometry.low_group,
                    geometry.low_group);
        if (geometry.high_group > 0) {
            std::memcpy(out.high.data() + g * geometry.high_group,
                        geometry.payload + geometry.high_offset +
                            source_group * geometry.high_group,
                        geometry.high_group);
        }
        std::memcpy(out.scales.data() + g * 2,
                    geometry.payload + geometry.scale_offset + source_group * 2, 2);
    }
    return out;
}

std::vector<std::byte> download(const void* device_ptr, std::size_t bytes, cudaStream_t stream) {
    std::vector<std::byte> host(bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device_ptr, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

bool bytes_equal(const std::vector<std::byte>& a, const std::vector<std::byte>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0;
}

// Compare one destination row of a materialized row-split tensor against host expectations.
// The destination geometry is computed from the rank's slice (rows and columns after slicing).
void check_row_split_row(const Reader& reader, RankLoad& rank, const char* name,
                         NumericFormat format, std::uint64_t destination_rows,
                         std::uint64_t destination_columns, std::uint64_t destination_row,
                         std::uint64_t source_row, std::uint64_t first_source_group,
                         std::uint64_t group_count, const std::string& label) {
    // K-halves keep the group run of the source row: destination group index of the first
    // checked group is the same offset inside the destination row.
    (void)destination_rows;
    const PlaneBytes expected =
        host_row_groups(reader, name, format, source_row, first_source_group, group_count);

    const RowSplitGeometryLite source_geometry = geometry_of(reader, name, format);
    const std::uint64_t destination_groups =
        (destination_columns + 127) / 128 * 128 /
        (format == NumericFormat::W8G32_F16S ? 32 : 64);
    const std::uint64_t high_group = source_geometry.high_group;
    const std::uint64_t total_destination_groups = destination_rows * destination_groups;

    const std::byte* base = static_cast<const std::byte*>(
        rank.materialized.device_data(ninfer::artifact::ObjectHandle{object_index(reader, name)}));
    const std::byte* codes = base;
    std::uint64_t offset   = total_destination_groups * 32;
    const std::byte* high  = high_group > 0 ? base + (offset + 255) / 256 * 256 : nullptr;
    if (high_group > 0) {
        offset = (offset + 255) / 256 * 256 + (total_destination_groups * high_group + 255) / 256 * 256;
    }
    const std::byte* scales = base + offset;
    const std::uint64_t group_offset =
        destination_row * destination_groups + first_source_group;

    CUDA_CHECK(cudaSetDevice(rank.device->device));
    const std::vector<std::byte> device_codes = download(
        codes + (group_offset * 32), expected.codes.size(), rank.device->load_stream);
    const std::vector<std::byte> device_scales =
        download(scales + group_offset * 2, expected.scales.size(), rank.device->load_stream);
    check(bytes_equal(device_codes, expected.codes), label + " codes");
    check(bytes_equal(device_scales, expected.scales), label + " scales");
    if (high_group > 0) {
        const std::vector<std::byte> device_high = download(
            high + group_offset * high_group, expected.high.size(), rank.device->load_stream);
        check(bytes_equal(device_high, expected.high), label + " high bits");
    }
}

// GDN convolution [4,10240] BF16 column gather: the rank's destination channel c of tap k must
// hold the source channel its [Q_rank; K_rank; V_rank] projection order names. Pins the shard
// table against regressing to a contiguous column span, which cross-wires the K/V conv channels.
void check_conv_gather(const Reader& reader, RankLoad& rank, const char* name, std::uint8_t r) {
    const auto* object = reader.find(name);
    require(object != nullptr, "convolution tensor is missing from the artifact");
    const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(object);
    require(tensor != nullptr && tensor->shape.size() == 2 && tensor->shape[0] == 4 &&
                tensor->shape[1] == 10240,
            "convolution tensor is not the stored [4,10240] BF16 matrix");
    const auto payload = reader.payload(*object);
    const std::uint16_t* source = reinterpret_cast<const std::uint16_t*>(payload.data.data());

    const auto channel_map = [r](std::uint64_t c) -> std::uint64_t {
        if (c < 1024) { return c + r * 1024; }              // Q rank block
        if (c < 2048) { return 2048 + r * 1024 + (c - 1024); } // K rank block
        return 4096 + r * 3072 + (c - 2048);               // V rank block
    };

    const std::byte* base = static_cast<const std::byte*>(
        rank.materialized.device_data(ninfer::artifact::ObjectHandle{object_index(reader, name)}));
    CUDA_CHECK(cudaSetDevice(rank.device->device));
    for (const std::uint64_t channel : {0ULL, 1ULL, 1023ULL, 1024ULL, 2047ULL, 2048ULL, 5119ULL}) {
        const std::uint64_t mapped = channel_map(channel);
        for (std::uint64_t tap = 0; tap < 4; ++tap) {
            const std::uint16_t expected = source[tap * 10240 + mapped];
            std::uint16_t actual         = 0;
            CUDA_CHECK(cudaMemcpyAsync(&actual, base + (tap * 5120 + channel) * 2, sizeof(actual),
                                       cudaMemcpyDeviceToHost, rank.device->load_stream));
            CUDA_CHECK(cudaStreamSynchronize(rank.device->load_stream));
            check(expected == actual,
                  "gdn convolution gather rank" + std::to_string(r) + " tap " +
                      std::to_string(tap) + " channel " + std::to_string(channel));
        }
    }
}

} // namespace

int run() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real artifact not found: " << path << '\n';
        return 77;
    }
    int device_count        = 0;
    const cudaError_t count = cudaGetDeviceCount(&device_count);
    if (count != cudaSuccess || device_count < 2) {
        std::cerr << "skip: two CUDA devices are required\n";
        return 77;
    }

    Reader reader(path);
    const WeightsProfile profile = Package::resolve_weights(reader.identity());

    // The single-GPU plan must be unchanged by the slice plumbing: whole-tensor placements only,
    // and each placement's byte count still equals the stored descriptor bytes.
    std::uint64_t single_capacity = 0;
    {
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan single = bind_artifact(binder, profile, text_features(), 0, false);
        require(single.materialization.host_objects.size() == 6,
                "single-GPU plan lost its frontend resources");
        for (const auto& placement : single.materialization.device_objects) {
            require(placement.slice.kind == ninfer::artifact::TensorSliceKind::Whole,
                    "single-GPU plan must place whole tensors only");
            const auto& object = reader.objects()[placement.object.index];
            require(ninfer::artifact::object_bytes(object) == placement.bytes,
                    "single-GPU placement bytes diverged from the stored tensor");
        }
        single_capacity = single.materialization.device_capacity_bytes;
    }

    RankLoad rank0 = load_rank(reader, profile, 0, 0);
    RankLoad rank1 = load_rank(reader, profile, 1, 1);

    const double r0 = static_cast<double>(rank0.arena_bytes) / static_cast<double>(single_capacity);
    const double r1 = static_cast<double>(rank1.arena_bytes) / static_cast<double>(single_capacity);
    std::cout << "arena bytes: rank0=" << rank0.arena_bytes << " rank1=" << rank1.arena_bytes
              << " single=" << single_capacity << " ratios=" << r0 << "/" << r1 << '\n';
    check(r0 >= 0.42 && r0 <= 0.52, "rank 0 arena bytes outside the expected band");
    check(r1 >= 0.42 && r1 <= 0.52, "rank 1 arena bytes outside the expected band");

    // ---- Byte-exact spot checks (D2H vs host extraction from the artifact payload) ----------

    // embedding [248320,5120] W8, vocab half rows: rank0 [0,124160), rank1 [124160,248320).
    check_row_split_row(reader, rank0, "text/token_embedding", NumericFormat::W8G32_F16S, 124160,
                        5120, 100000, 100000, 0, 160, "embedding rank0 row 100000");
    check_row_split_row(reader, rank1, "text/token_embedding", NumericFormat::W8G32_F16S, 124160,
                        5120, 200000 - 124160, 200000, 0, 160, "embedding rank1 row 200000");
    // layer 3 o_proj [5120,6144] Q5 K-half on rank 0: columns [0,3072) -> destination groups
    // [0,48); row 17 group 5 reads source row 17, source group 5.
    check_row_split_row(reader, rank0, "text/layers/3/attention/output", NumericFormat::Q5G64_F16S,
                        5120, 3072, 17, 17, 5, 1, "layer3 o_proj K-half rank0 row17 group5");

    // layer 7 mlp gate_up [34816,5120] Q4 repack on rank 0: destination rows [0,8704) are gate
    // source rows [0,8704), rows [8704,17408) are up source rows [17408,26112). Destination row
    // 9000 is up source row 17408 + (9000 - 8704) = 17704.
    check_row_split_row(reader, rank0, "text/layers/7/mlp/gate_up", NumericFormat::Q4G64_F16S,
                        17408, 5120, 9000, 17408 + 9000 - 8704, 0, 80, "layer7 gate_up rank0 row9000");

    // GDN layer 0 value_z [12288,5120] Q5 on rank 1: destination rows [0,3072) are value source
    // rows [3072,6144). Destination row 100 is source row 3172.
    check_row_split_row(reader, rank1, "text/layers/0/gdn/value_z", NumericFormat::Q5G64_F16S, 6144,
                        5120, 100, 3072 + 100, 0, 80, "gdn0 value_z rank1 row100");

    // GDN layer 0 convolution column gather on both ranks (see check_conv_gather).
    check_conv_gather(reader, rank0, "text/layers/0/gdn/convolution", 0);
    check_conv_gather(reader, rank1, "text/layers/0/gdn/convolution", 1);

    // lm_head [248320,5120] W8 rank 0 last row 124159.
    check_row_split_row(reader, rank0, "text/output_head", NumericFormat::W8G32_F16S, 124160, 5120,
                        124159, 124159, 0, 160, "lm_head rank0 row124159");

    // MTP input_projection [5120,10240] W8 K-half on rank 0: columns [0,5120) -> groups [0,160).
    check_row_split_row(reader, rank0, "mtp/input_projection", NumericFormat::W8G32_F16S, 5120,
                        5120, 9, 9, 3, 5, "mtp input_projection K-half rank0 row9 group3");

    std::cout << (failures == 0 ? "ok\n" : "failures\n");
    return failures == 0 ? 0 : 1;
}

int main() {
    std::cout << std::unitbuf;
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
