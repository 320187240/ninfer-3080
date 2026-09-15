#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

// One rank's slice of a stored tensor, expressed against the full artifact shape. Rows keeps the
// stored layout per plane and concatenates disjoint source row ranges; Columns keeps every row and
// gathers disjoint column ranges in order, densified at materialization time. Both are exact byte
// transforms.
enum class TensorSliceKind : std::uint8_t {
    Whole,
    Rows,
    Columns,
};

struct TensorSliceRange {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
};

struct TensorSlice {
    TensorSliceKind kind                          = TensorSliceKind::Whole;
    std::array<TensorSliceRange, 4> rows          = {};
    std::array<TensorSliceRange, 4> column_ranges = {};

    [[nodiscard]] constexpr std::size_t used_row_ranges() const noexcept {
        std::size_t used = 0;
        for (const TensorSliceRange& range : rows) {
            if (range.count == 0) { break; }
            ++used;
        }
        return used;
    }

    [[nodiscard]] constexpr std::uint64_t row_count() const noexcept {
        std::uint64_t total = 0;
        for (const TensorSliceRange& range : rows) { total += range.count; }
        return total;
    }

    [[nodiscard]] constexpr std::size_t used_column_ranges() const noexcept {
        std::size_t used = 0;
        for (const TensorSliceRange& range : column_ranges) {
            if (range.count == 0) { break; }
            ++used;
        }
        return used;
    }

    [[nodiscard]] constexpr std::uint64_t column_count() const noexcept {
        std::uint64_t total = 0;
        for (const TensorSliceRange& range : column_ranges) { total += range.count; }
        return total;
    }
};

std::uint64_t sliced_tensor_encoded_size(const TensorSlice& slice, StorageLayout layout,
                                         NumericFormat format,
                                         std::span<const std::uint64_t> shape);

// Per-plane source/destination layout of a slice: each plane covers `rows` source rows of
// `segment_bytes` at `segment_offset` within every `source_row_stride` bytes, densified into
// `destination_row_stride`. Rows-slice planes still carry full-row segments; the materializer
// coalesces their row ranges into plain copies.
struct TensorSlicePlane {
    std::uint64_t source_offset           = 0;
    std::uint64_t destination_offset      = 0;
    std::uint64_t rows                    = 0;
    std::uint64_t source_row_stride       = 0;
    std::uint64_t destination_row_stride  = 0;
    std::uint64_t segment_offset          = 0;
    std::uint64_t segment_bytes           = 0;
};

std::vector<TensorSlicePlane> tensor_slice_planes(const TensorSlice& slice, StorageLayout layout,
                                                  NumericFormat format,
                                                  std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
