// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace din::seedvr
{
// Native FP16 VAE. Device I/O is contiguous [1,C,T,H,W], on the supplied stream.
// One instance per stream/thread. Weights are prepared from ONNX by prepare_native_vae.py.
// Construction builds plans and storage for the explicitly configured shapes.
// First call per direction warms plans and captures a CUDA Graph. Later
// calls enqueue copies + replay without internal synchronization or allocation.
// Device I/O must remain valid until the supplied non-default stream completes.
// Fixed sessions (including graph-owned storage) persist until destruction.
class NativeVae
{
public:
    struct Shape
    {
        int frames, height, width;
    };
    struct Shapes
    {
        std::optional<Shape> encoder;  // RGB input, including tile dimensions.
        std::optional<Shape> decoder;  // Latent input, including tile dimensions.
    };
    explicit NativeVae(const std::filesystem::path& weights, cudaStream_t stream, const Shapes& shapes,
                       size_t workspace_limit = 256ull << 20, bool autotune = false,
                       const std::filesystem::path& tuning_cache = {}, int tuning_candidates = 5);
    ~NativeVae();
    NativeVae(const NativeVae&) = delete;
    NativeVae& operator=(const NativeVae&) = delete;
    void Encode(const void* image, void* latent);
    void Decode(const void* latent, void* image);
    // Release synchronizes this instance's stream; virtual addresses and graphs
    // survive. Commit (or the next Run) restores backing without recapturing.
    // Working contents are discarded; persistent I/O allocations remain intact.
    // Calls on an instance must be serialized.
    void ReleaseWorkingMemory();
    void CommitWorkingMemory();
    struct WorkingMemoryStats
    {
        size_t encoder_bytes, decoder_bytes, shared_bytes, resident_bytes;
        // Separate, persistent cudaMallocAsync allocations (not shared VMM).
        size_t encoder_io_bytes, decoder_io_bytes;
    };
    WorkingMemoryStats GetWorkingMemoryStats() const;
    // Physical backing high-water mark, counting all virtual aliases once;
    // includes scratch, excludes persistent staging I/O, weights and caller I/O.
    size_t PeakWorkingBytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace din::seedvr
