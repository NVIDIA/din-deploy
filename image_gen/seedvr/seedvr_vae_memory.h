// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// Internal VAE working-memory planner. Encoder and decoder execute serially on
// one stream and map separate, stable virtual views onto the same physical heap.
namespace din::seedvr::working_memory
{
inline void check(cudaError_t status, const char* api,
                  const std::source_location location = std::source_location::current())
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(api) + " failed: " + cudaGetErrorName(status) + " (" +
                                 std::to_string(int(status)) + "): " + cudaGetErrorString(status) + " at " +
                                 location.file_name() + ":" + std::to_string(location.line()));
}
inline void check(CUresult status, const char* api,
                  const std::source_location location = std::source_location::current())
{
    if (status == CUDA_SUCCESS)
        return;
    const char* message = nullptr;
    const char* name = nullptr;
    cuGetErrorString(status, &message);
    cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(api) + " failed: " + (name ? name : "CUDA_ERROR_UNKNOWN") + " (" +
                             std::to_string(int(status)) + "): " + (message ? message : "unknown driver error") +
                             " at " + location.file_name() + ":" + std::to_string(location.line()));
}
inline std::string memory_snapshot()
{
    size_t free = 0, total = 0;
    const auto status = cudaMemGetInfo(&free, &total);
    if (status != cudaSuccess)
        return std::string("cudaMemGetInfo unavailable: ") + cudaGetErrorName(status) + " (" +
               std::to_string(int(status)) + ")";
    if (!total || free > total)
        return "cudaMemGetInfo invalid sample: free_bytes=" + std::to_string(free) +
               " total_bytes=" + std::to_string(total);
    return "device_free_bytes=" + std::to_string(free) + " device_total_bytes=" + std::to_string(total) +
           " (device-wide snapshot, not a reservation guarantee)";
}
inline thread_local unsigned cleanup_warnings = 0;
inline void cleanup(CUresult status, const char* api) noexcept
{
    if (status == CUDA_SUCCESS)
        return;
    if (cleanup_warnings++ < 8)
        std::fprintf(stderr, "[VAE cleanup] %s returned CUDA status %d; preserving primary failure\n", api,
                     int(status));
    else if (cleanup_warnings == 9)
        std::fprintf(stderr, "[VAE cleanup] further cleanup errors suppressed\n");
}
inline size_t align_up(size_t n, size_t alignment)
{
    if (n > SIZE_MAX - (alignment - 1))
        throw std::overflow_error("VAE allocation overflow");
    return (n + alignment - 1) / alignment * alignment;
}

struct Planner
{
    struct Allocation
    {
        size_t bytes, begin, end = SIZE_MAX, offset = 0;
    };
    size_t extent = 0;
    size_t clock = 0;
    bool finalized = false;
    std::vector<Allocation> allocations;
    std::map<size_t, size_t> tokens;
    // Tokens identify allocations while operations capture their virtual pointers.
    // Physical offsets are assigned only after every temporary lifetime is known.
    size_t acquire(size_t bytes)
    {
        if (finalized)
            throw std::logic_error("Allocation after VAE placement finalization");
        const size_t offset = extent;
        if (bytes > SIZE_MAX - extent)
            throw std::overflow_error("VAE arena overflow");
        if (bytes)
        {
            tokens.emplace(offset, allocations.size());
            allocations.push_back({bytes, ++clock});
        }
        extent += bytes;
        return offset;
    }
    void release(size_t offset, size_t bytes)
    {
        if (!bytes)
            return;
        auto& a = allocations.at(tokens.at(offset));
        if (finalized || a.bytes != bytes || a.end != SIZE_MAX)
            throw std::logic_error("Invalid VAE allocation lifetime");
        a.end = ++clock;
    }
    static bool overlaps(const Allocation& a, const Allocation& b)
    {
        return a.begin < b.end && b.begin < a.end;
    }
    void finalize()
    {
        if (finalized)
            throw std::logic_error("VAE placement already finalized");
        for (const auto& a : allocations)
            if (a.end == SIZE_MAX)
                throw std::logic_error("Unclosed VAE temporary lifetime");
        std::vector<size_t> order(allocations.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b)
                         {
                             return allocations[a].bytes > allocations[b].bytes;
                         });
        std::vector<size_t> placed;
        extent = 0;
        for (const auto i : order)
        {
            auto& a = allocations[i];
            std::vector<std::pair<size_t, size_t>> occupied;
            for (const auto j : placed)
            {
                const auto& b = allocations[j];
                if (overlaps(a, b))
                    occupied.emplace_back(b.offset, b.offset + b.bytes);
            }
            std::sort(occupied.begin(), occupied.end());
            size_t offset = 0;
            for (auto [begin, end] : occupied)
            {
                if (offset <= begin && a.bytes <= begin - offset)
                    break;
                offset = std::max(offset, end);
            }
            if (a.bytes > SIZE_MAX - offset)
                throw std::overflow_error("VAE placement overflow");
            a.offset = offset;
            extent = std::max(extent, offset + a.bytes);
            placed.push_back(i);
        }
        // Validate independently before any physical mapping or GPU execution.
        for (size_t i = 0; i < allocations.size(); ++i)
            for (size_t j = i + 1; j < allocations.size(); ++j)
            {
                const auto& a = allocations[i];
                const auto& b = allocations[j];
                if (overlaps(a, b) && a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes)
                    throw std::logic_error("Overlapping live VAE allocations");
            }
        finalized = true;
    }
    size_t placement(size_t token) const
    {
        if (!finalized)
            throw std::logic_error("VAE placement is not finalized");
        return allocations.at(tokens.at(token)).offset;
    }
};

class View;
class Arena
{
public:
    CUmemAllocationProp prop{};
    size_t granularity = 0, capacity = 0, resident = 0, peak = 0;
    std::vector<View*> views;
    struct Chunk
    {
        size_t offset, bytes;
        CUmemGenericAllocationHandle handle;
    };
    std::vector<Chunk> chunks;
    Arena()
    {
        int device;
        check(cudaGetDevice(&device), "cudaGetDevice");
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
    }
    ~Arena()
    {
        release();
    }
    void commit();
    void update_capacity();
    // Caller must first drain the stream. Virtual reservations remain intact.
    void release();
};

class View
{
public:
    Arena& arena;
    CUdeviceptr base = 0;
    size_t span = 0, offset = 0, bytes = 0, prefix = 0;
    struct Region
    {
        size_t virtual_offset, physical_offset, bytes;
        bool read_only;
    };
    std::vector<Region> regions;
    std::vector<std::pair<CUdeviceptr, size_t>> mapped;
    View(Arena& a, size_t physical_offset, size_t physical_bytes, size_t frame_alias = 0)
        : arena(a)
        , offset(physical_offset)
        , bytes(physical_bytes)
        , prefix(2 * frame_alias)
    {
        if (!bytes)
            return;
        span = bytes + prefix;
        if (span < bytes)
            throw std::overflow_error("VAE virtual span overflow");
        if (frame_alias)
        {
            regions.push_back({0, offset, frame_alias, true});
            regions.push_back({frame_alias, offset, frame_alias, true});
        }
        regions.push_back({prefix, offset, bytes, false});
        const auto reserve_status = cuMemAddressReserve(&base, span, arena.granularity, 0, 0);
        try
        {
            check(reserve_status, "cuMemAddressReserve");
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string(e.what()) + "; virtual_span_bytes=" + std::to_string(span) +
                                     " alignment_bytes=" + std::to_string(arena.granularity));
        }
        try
        {
            arena.views.push_back(this);
        }
        catch (...)
        {
            cleanup(cuMemAddressFree(base, span), "cuMemAddressFree");
            throw;
        }
        arena.capacity = std::max(arena.capacity, offset + bytes);
    }
    ~View()
    {
        unmap();
        if (base)
            cleanup(cuMemAddressFree(base, span), "cuMemAddressFree");
        auto& v = arena.views;
        v.erase(std::remove(v.begin(), v.end(), this), v.end());
    }
    void* data() const
    {
        return reinterpret_cast<void*>(base + prefix);
    }
    void relocate(size_t physical_offset)
    {
        if (!mapped.empty() || !arena.chunks.empty() || physical_offset % arena.granularity)
            throw std::logic_error("Invalid VAE relocation after mapping or unaligned offset");
        for (auto& region : regions)
            region.physical_offset = physical_offset + (region.physical_offset - offset);
        offset = physical_offset;
    }
    void unmap()
    {
        for (auto [address, length] : mapped)
            cleanup(cuMemUnmap(address, length), "cuMemUnmap");
        mapped.clear();
    }
    void map()
    {
        for (const auto& region : regions)
        {
            for (const auto& chunk : arena.chunks)
            {
                if (chunk.offset < region.physical_offset || chunk.offset >= region.physical_offset + region.bytes)
                    continue;
                const auto address = base + region.virtual_offset + chunk.offset - region.physical_offset;
                // All region boundaries are chunk boundaries; handle offset is
                // always zero (required by CUDA VMM). No duplicate physical bytes.
                try
                {
                    check(cuMemMap(address, chunk.bytes, 0, chunk.handle, 0), "cuMemMap");
                    mapped.emplace_back(address, chunk.bytes);
                    CUmemAccessDesc access{};
                    access.location = arena.prop.location;
                    access.flags =
                        region.read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    check(cuMemSetAccess(address, chunk.bytes, &access, 1), "cuMemSetAccess");
                }
                catch (const std::exception& e)
                {
                    std::ostringstream context;
                    context << e.what() << "; device=" << arena.prop.location.id << " virtual_address=0x" << std::hex
                            << address << std::dec << " mapping_bytes=" << chunk.bytes
                            << " physical_offset_bytes=" << chunk.offset << " view_span_bytes=" << span
                            << " access=" << (region.read_only ? "read-only causal alias" : "read-write");
                    throw std::runtime_error(context.str());
                }
            }
        }
    }
};

inline void Arena::update_capacity()
{
    if (resident || !chunks.empty())
        throw std::logic_error("Cannot replan a committed VAE arena");
    capacity = 0;
    for (const auto* view : views)
        if (view->bytes)
            capacity = std::max(capacity, view->offset + view->bytes);
}

inline void Arena::commit()
{
    if (resident || !capacity)
        return;
    std::vector<size_t> cuts{0, capacity};
    for (auto* view : views)
        for (const auto& region : view->regions)
        {
            cuts.push_back(region.physical_offset);
            cuts.push_back(region.physical_offset + region.bytes);
        }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    chunks.reserve(cuts.size());
    cleanup_warnings = 0;
    std::cerr << "[VAE VMM] committing shared backing: required_bytes=" << capacity << " chunks=" << cuts.size() - 1
              << " virtual_views=" << views.size() << " device=" << prop.location.id << "; " << memory_snapshot()
              << std::endl;
    size_t created_bytes = 0;
    try
    {
        for (size_t i = 1; i < cuts.size(); ++i)
        {
            CUmemGenericAllocationHandle handle;
            const size_t bytes = cuts[i] - cuts[i - 1];
            const auto status = cuMemCreate(&handle, bytes, &prop, 0);
            try
            {
                check(status, "cuMemCreate");
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(std::string(e.what()) + "; chunk_bytes=" + std::to_string(bytes) +
                                         " physical_offset_bytes=" + std::to_string(cuts[i - 1]));
            }
            chunks.push_back({cuts[i - 1], bytes, handle});
            created_bytes += bytes;
        }
        for (auto* view : views)
        {
            // Reserve tracking before mapping so allocation failure cannot
            // leave an untracked live mapping.
            view->mapped.reserve(chunks.size() * view->regions.size());
            view->map();
        }
        resident = capacity;
        peak = std::max(peak, resident);
        std::cerr << "[VAE VMM] shared backing committed: resident_bytes=" << resident << std::endl;
    }
    catch (const std::exception& e)
    {
        const auto primary = std::string(e.what()) + "; shared_required_bytes=" + std::to_string(capacity) +
                             " physical_created_bytes=" + std::to_string(created_bytes) +
                             " chunks_created=" + std::to_string(chunks.size()) + "; " + memory_snapshot();
        std::cerr << "[VAE VMM] primary commit failure before cleanup: " << primary << std::endl;
        release();
        throw std::runtime_error(primary);
    }
    catch (...)
    {
        release();
        throw;
    }
}

inline void Arena::release()
{
    for (auto* view : views)
        view->unmap();
    for (const auto& chunk : chunks)
        cleanup(cuMemRelease(chunk.handle), "cuMemRelease");
    chunks.clear();
    resident = 0;
}
}  // namespace din::seedvr::working_memory
