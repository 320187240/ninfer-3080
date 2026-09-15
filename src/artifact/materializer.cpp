#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) { throw ArtifactError(label); }
    return a * b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

// A column-sliced placement: its bytes are gathered per row out of sequentially read source planes
// and densified into the destination row stride.
struct GatherObject {
    std::byte* destination = nullptr;
    std::uint64_t payload_offset = 0;
    std::vector<TensorSlicePlane> planes;
};

// Accumulates densified slice bytes in the pinned slots and uploads contiguous destination runs.
class GatherWindow {
public:
    GatherWindow(std::span<Slot* const> slots, cudaStream_t stream)
        : slots_(slots.begin(), slots.end()), stream_(stream) {}

    void append(std::byte* destination, const std::byte* source, std::size_t bytes) {
        while (bytes > 0) {
            if (current_ == nullptr || run_destination_ + run_bytes_ != destination) {
                if (current_ != nullptr) { flush(); }
                begin_run(destination);
            }
            const std::size_t capacity = current_->buffer.size();
            const std::size_t take     = std::min(bytes, capacity - run_bytes_);
            std::memcpy(static_cast<std::byte*>(current_->buffer.data()) + run_bytes_, source,
                        take);
            run_bytes_ += take;
            source += take;
            destination += take;
            bytes -= take;
            if (run_bytes_ == capacity) { flush(); }
        }
    }

    void flush() {
        if (run_bytes_ == 0) { return; }
        CUDA_CHECK(cudaMemcpyAsync(run_destination_,
                                   static_cast<std::byte*>(current_->buffer.data()), run_bytes_,
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaEventRecord(current_->event, stream_));
        current_->pending = true;
        uploaded_ += run_bytes_;
        run_bytes_ = 0;
    }

    [[nodiscard]] std::uint64_t uploaded() const noexcept { return uploaded_; }

private:
    void begin_run(std::byte* destination) {
        current_        = slots_[next_slot_++ % slots_.size()];
        current_->wait();
        run_destination_ = destination;
        run_bytes_       = 0;
    }

    std::vector<Slot*> slots_;
    cudaStream_t stream_;
    Slot* current_           = nullptr;
    std::size_t next_slot_   = 0;
    std::byte* run_destination_ = nullptr;
    std::size_t run_bytes_   = 0;
    std::uint64_t uploaded_  = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    std::vector<GatherObject> gathers;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const ObjectDescriptor& object = reader.objects().at(placement.object.index);
        const PayloadSpan payload      = reader.payload(object);
        const auto* tensor             = std::get_if<TensorDescriptor>(&object);
        if (tensor == nullptr) {
            throw ArtifactError("resource cannot be materialized as a device tensor");
        }
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        const std::span<const std::uint64_t> shape(tensor->shape.data(), tensor->shape.size());

        if (placement.slice.kind == TensorSliceKind::Whole) {
            if (payload.data.size() != placement.bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            ranges.push_back(CopyRange{
                .source_begin = payload.absolute_offset,
                .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = static_cast<std::byte*>(storage.data),
            });
            total =
                checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
            continue;
        }

        if (placement.slice.kind == TensorSliceKind::Columns) {
            GatherObject gather;
            gather.destination    = static_cast<std::byte*>(storage.data);
            gather.payload_offset = payload.absolute_offset;
            gather.planes         = tensor_slice_planes(placement.slice, tensor->layout,
                                                        tensor->format, shape);
            gathers.push_back(std::move(gather));
            total =
                checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
            continue;
        }

        // Rows slice: layout-native concatenation of disjoint whole-row runs. Every plane of
        // every range is one contiguous source run copied to the destination row offset.
        std::vector<TensorSlicePlane> planes =
            tensor_slice_planes(placement.slice, tensor->layout, tensor->format, shape);
        std::uint64_t destination_row = 0;
        for (const TensorSliceRange& range : placement.slice.rows) {
            if (range.count == 0) { break; }
            for (const TensorSlicePlane& plane : planes) {
                const std::uint64_t copy_bytes = checked_mul(
                    range.count, plane.source_row_stride, "tensor slice bytes overflow u64");
                const std::uint64_t source_begin = checked_add(
                    payload.absolute_offset + plane.source_offset,
                    checked_mul(range.begin, plane.source_row_stride,
                                "artifact tensor slice offset overflows u64"),
                    "artifact tensor source range overflows u64");
                ranges.push_back(CopyRange{
                    .source_begin = source_begin,
                    .source_end   = checked_add(source_begin, copy_bytes,
                                                "artifact tensor source range overflows u64"),
                    .destination  = static_cast<std::byte*>(storage.data) + plane.destination_offset +
                                    destination_row * plane.destination_row_stride,
                });
                total = checked_add(total, copy_bytes, "artifact tensor byte count overflows u64");
            }
            destination_row += range.count;
        }
    }
    if (ranges.empty() && gathers.empty()) {
        throw ArtifactError("materialization plan has no device tensors");
    }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::uint64_t staging_bytes =
        std::max(aligned_read_bytes, gathers.empty() ? 0 : 2 * Reader::direct_io_alignment);
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, staging_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (staging_bytes - 1) / slot_bytes));
    // Column slices densify through their own pinned slots; sharing the direct-read slots would
    // memcpy overlapping regions of one buffer.
    const std::size_t window_slot_count = gathers.empty() ? 0 : 2;
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count + window_slot_count);
    for (std::size_t i = 0; i < slot_count + window_slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    out.stats_.peak_staging_bytes =
        static_cast<std::uint64_t>(slot_bytes) * (slot_count + window_slot_count);

    std::vector<Slot*> slot_ptrs;
    slot_ptrs.reserve(slots.size());
    for (const auto& slot : slots) { slot_ptrs.push_back(slot.get()); }
    GatherWindow window(std::span<Slot* const>(slot_ptrs).subspan(slot_count), device.load_stream);

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");
            const std::byte* chunk_data = static_cast<const std::byte*>(slot.buffer.data());

            while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        chunk_data + static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, device.load_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
            slot.pending = true;

            const std::uint64_t done = copied + window.uploaded();
            if (progress != nullptr && progress->callback && done != last_published &&
                done < total) {
                last_published = done;
                progress->callback("weights", done, total);
            }
        }
    }

    // Column slices: one direct read per plane batch, aligned out to the direct-I/O boundary and
    // sized to whole source rows, so every densified segment is fully contained in the batch.
    for (const GatherObject& gather : gathers) {
        for (const TensorSlicePlane& plane : gather.planes) {
            const std::uint64_t plane_begin = gather.payload_offset + plane.source_offset;
            std::uint64_t row = 0;
            while (row < plane.rows) {
                Slot& slot = *slot_ptrs[next_slot++ % slot_count];
                slot.wait();
                const std::uint64_t alignment_slack = 2 * Reader::direct_io_alignment;
                const std::uint64_t batch_capacity =
                    slot_bytes > alignment_slack ? slot_bytes - alignment_slack : slot_bytes;
                const std::uint64_t batch_rows =
                    std::max<std::uint64_t>(1, batch_capacity / plane.source_row_stride);
                const std::uint64_t rows_now = std::min(batch_rows, plane.rows - row);
                const std::uint64_t batch_begin = plane_begin + row * plane.source_row_stride;
                const std::uint64_t batch_end =
                    batch_begin + rows_now * plane.source_row_stride;
                const std::uint64_t read_begin = align_down(batch_begin, alignment);
                const std::uint64_t read_end   = align_up(batch_end, alignment,
                                                          "artifact direct I/O span overflows u64");
                const auto request = static_cast<std::size_t>(read_end - read_begin);
                if (request > slot_bytes) {
                    throw ArtifactError("column-sliced tensor batch exceeds the staging slot");
                }
                const std::size_t bytes_read = reader.read_direct(
                    read_begin,
                    std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request));
                if (bytes_read < batch_end - read_begin) {
                    throw ArtifactError("direct artifact read ended before the planned tensor range");
                }
                out.stats_.file_bytes =
                    checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
                const std::byte* chunk = static_cast<const std::byte*>(slot.buffer.data());
                const std::size_t chunk_lead =
                    static_cast<std::size_t>(batch_begin - read_begin);
                for (std::uint64_t batch_row = 0; batch_row < rows_now; ++batch_row) {
                    const std::uint64_t source_row = row + batch_row;
                    window.append(
                        gather.destination + plane.destination_offset +
                            source_row * plane.destination_row_stride,
                        chunk + chunk_lead +
                            static_cast<std::size_t>(batch_row * plane.source_row_stride +
                                                     plane.segment_offset),
                        static_cast<std::size_t>(plane.segment_bytes));
                }
                row += rows_now;
                if (progress != nullptr && progress->callback) {
                    const std::uint64_t done = copied + window.uploaded();
                    if (done != last_published && done < total) {
                        last_published = done;
                        progress->callback("weights", done, total);
                    }
                }
            }
        }
    }
    window.flush();
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    const std::uint64_t uploaded = copied + window.uploaded();
    if (uploaded != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = uploaded;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", uploaded, total); }
    return out;
}

} // namespace ninfer::artifact
