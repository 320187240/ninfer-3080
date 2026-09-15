#include "artifact/reader.h"

#include <limits>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, or I32");
    }
}

} // namespace

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape) {
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        return block_scale_geometry(format, shape).encoded_bytes;
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

void validate_row_ranges(const TensorSlice& slice, std::uint64_t rows) {
    if (slice.used_row_ranges() == 0) {
        throw ArtifactError("tensor row slice selects no rows");
    }
    std::uint64_t cursor = 0;
    for (const TensorSliceRange& range : slice.rows) {
        if (range.count == 0) { break; }
        if (range.begin < cursor) {
            throw ArtifactError("tensor row slice ranges overlap or are out of order");
        }
        if (range.begin > rows || range.count > rows - range.begin) {
            throw ArtifactError("tensor row slice range is outside the stored shape");
        }
        cursor = range.begin + range.count;
    }
}

void validate_column_ranges(const TensorSlice& slice, std::uint64_t columns) {
    if (slice.used_column_ranges() == 0) {
        throw ArtifactError("tensor column slice selects no columns");
    }
    std::uint64_t cursor = 0;
    for (const TensorSliceRange& range : slice.column_ranges) {
        if (range.count == 0) { break; }
        if (range.begin < cursor) {
            throw ArtifactError("tensor column slice ranges overlap or are out of order");
        }
        if (range.begin > columns || range.count > columns - range.begin) {
            throw ArtifactError("tensor column slice range is outside the stored shape");
        }
        cursor = range.begin + range.count;
    }
}

std::uint64_t contiguous_row_bytes(std::span<const std::uint64_t> shape) {
    std::uint64_t elements = 1;
    for (std::size_t dim = 1; dim < shape.size(); ++dim) {
        elements = checked_mul(elements, shape[dim], "tensor row element count");
    }
    return elements;
}

std::uint64_t sliced_tensor_encoded_size(const TensorSlice& slice, StorageLayout layout,
                                         NumericFormat format,
                                         std::span<const std::uint64_t> shape) {
    if (shape.empty()) { throw ArtifactError("tensor slice requires a positive shape"); }
    switch (slice.kind) {
    case TensorSliceKind::Whole:
        return tensor_encoded_size(layout, format, shape);
    case TensorSliceKind::Rows:
        validate_row_ranges(slice, shape[0]);
        if (layout == StorageLayout::ContiguousLeV1) {
            const auto row_bytes = checked_mul(contiguous_row_bytes(shape),
                                               direct_word_bytes(format), "tensor row bytes");
            return checked_mul(slice.row_count(), row_bytes, "sliced tensor encoded size");
        }
        if (layout == StorageLayout::RowSplitK128V1) {
            const std::array<std::uint64_t, 2> sliced = {slice.row_count(), shape[1]};
            return row_split_geometry(format, sliced).encoded_bytes;
        }
        break;
    case TensorSliceKind::Columns:
        if (shape.size() != 2 || slice.column_count() == 0) {
            throw ArtifactError("tensor column slice requires a positive rank-two shape");
        }
        validate_column_ranges(slice, shape[1]);
        if (layout == StorageLayout::ContiguousLeV1) {
            const auto elements = checked_mul(shape[0], slice.column_count(),
                                              "sliced tensor element count");
            return checked_mul(elements, direct_word_bytes(format), "sliced tensor encoded size");
        }
        if (layout == StorageLayout::RowSplitK128V1) {
            // The physical group run must land on whole groups of every range and on the
            // row-split K padding boundary, so the destination stays a dense row-split tensor.
            for (const TensorSliceRange& range : slice.column_ranges) {
                if (range.count == 0) { break; }
                if (range.begin % kKAlignment != 0 || range.count % kKAlignment != 0) {
                    throw ArtifactError(
                        "row-split tensor column slice is not aligned to the K padding boundary");
                }
            }
            const std::array<std::uint64_t, 2> sliced = {shape[0], slice.column_count()};
            return row_split_geometry(format, sliced).encoded_bytes;
        }
        break;
    }
    throw ArtifactError("tensor layout does not support this slice kind");
}

std::vector<TensorSlicePlane> tensor_slice_planes(const TensorSlice& slice, StorageLayout layout,
                                                  NumericFormat format,
                                                  std::span<const std::uint64_t> shape) {
    // Validates the slice and returns the per-plane source/destination layout the materializer
    // walks. Every plane of a Rows slice is a set of whole-row runs (segment covers the row); a
    // Columns plane is a per-row segment gather densified into the destination row stride.
    (void)sliced_tensor_encoded_size(slice, layout, format, shape);
    std::vector<TensorSlicePlane> planes;

    if (layout == StorageLayout::ContiguousLeV1) {
        const std::uint64_t row_bytes =
            checked_mul(contiguous_row_bytes(shape), direct_word_bytes(format), "tensor row bytes");
        if (slice.kind == TensorSliceKind::Columns) {
            // One plane per column range: each source row contributes its segment, densified at
            // the range's prefix offset inside the destination row stride.
            const std::uint64_t word = direct_word_bytes(format);
            std::uint64_t destination_offset = 0;
            for (const TensorSliceRange& range : slice.column_ranges) {
                if (range.count == 0) { break; }
                TensorSlicePlane plane{};
                plane.rows                   = shape[0];
                plane.source_row_stride      = row_bytes;
                plane.destination_row_stride = slice.column_count() * word;
                plane.destination_offset     = destination_offset;
                plane.segment_offset         = range.begin * word;
                plane.segment_bytes          = range.count * word;
                planes.push_back(plane);
                destination_offset += range.count * word;
            }
            return planes;
        }
        TensorSlicePlane plane{};
        plane.rows                   = shape[0];
        plane.source_row_stride      = row_bytes;
        plane.destination_row_stride = row_bytes;
        plane.segment_bytes          = row_bytes;
        planes.push_back(plane);
        return planes;
    }

    if (layout != StorageLayout::RowSplitK128V1) {
        throw ArtifactError("tensor layout does not support this slice kind");
    }
    const auto format_geometry = quant_geometry(format);
    const RowSplitGeometry source = row_split_geometry(format, shape);
    const std::array<std::uint64_t, 2> destination_shape = {
        slice.kind == TensorSliceKind::Columns ? shape[0] : slice.row_count(),
        slice.kind == TensorSliceKind::Columns ? slice.column_count() : shape[1]};
    const RowSplitGeometry destination = row_split_geometry(format, destination_shape);

    struct RowSplitPlane {
        std::uint64_t source_offset;
        std::uint64_t destination_offset;
        std::uint64_t bytes_per_group;
    };
    const RowSplitPlane layout_planes[] = {
        {0, 0, format_geometry.base_bytes_per_group},
        {source.high_plane_offset, destination.high_plane_offset,
         format_geometry.high_bytes_per_group},
        {source.scale_plane_offset, destination.scale_plane_offset, 2},
    };
    if (slice.kind == TensorSliceKind::Columns) {
        // One plane per (range, physical plane): the range's groups land at its group prefix
        // inside the densified destination row.
        std::uint64_t destination_group = 0;
        for (const TensorSliceRange& range : slice.column_ranges) {
            if (range.count == 0) { break; }
            const std::uint64_t range_groups = range.count / source.group_size;
            for (const RowSplitPlane& plane : layout_planes) {
                if (plane.bytes_per_group == 0) { continue; }
                TensorSlicePlane out{};
                out.rows                   = shape[0];
                out.source_offset          = plane.source_offset;
                out.destination_offset     = plane.destination_offset +
                                         destination_group * plane.bytes_per_group;
                out.source_row_stride      = source.groups_per_row * plane.bytes_per_group;
                out.destination_row_stride = destination.groups_per_row * plane.bytes_per_group;
                out.segment_offset = (range.begin / source.group_size) * plane.bytes_per_group;
                out.segment_bytes  = range_groups * plane.bytes_per_group;
                planes.push_back(out);
            }
            destination_group += range_groups;
        }
        return planes;
    }
    for (const RowSplitPlane& plane : layout_planes) {
        if (plane.bytes_per_group == 0) { continue; }
        TensorSlicePlane out{};
        out.rows                   = shape[0];
        out.source_offset          = plane.source_offset;
        out.destination_offset     = plane.destination_offset;
        out.source_row_stride      = source.groups_per_row * plane.bytes_per_group;
        out.destination_row_stride = source.groups_per_row * plane.bytes_per_group;
        out.segment_bytes          = out.source_row_stride;
        planes.push_back(out);
    }
    return planes;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
