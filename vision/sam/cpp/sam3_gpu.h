// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sam3_pipeline.h"
#include "../../common/device_frame.h"
namespace din::sam3 {
// Lossless low-resolution logits, validity and presence scores for offline storage.
// Full source frames never leave the GPU through this API.
struct SavedMask {
    static constexpr size_t Values = 16 * 288 * 288 + 32;
    int width = 0, height = 0;
    bool temporal = true;
    std::vector<float> values;
};
// Model-free compositor. Retains pinned staging and device allocations; CUDA
// callers receive an event dependency, with no stream wait on the returning host.
class SavedMaskRenderer {
public:
    SavedMaskRenderer();
    ~SavedMaskRenderer();
    void Render(const SavedMask&, const vision::DeviceRgbaView&,
                const vision::DeviceOutputView&, void* host_stream, int selected = 0, bool labels = false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Serialized API. Submit/Render keep frame/mask tensors on device; Export is an
// explicit offline readback boundary. Non-null host streams
// receive an event wait before return; borrowed frames live until that wait completes.
// Cold initialization, prompt/shape changes and destruction may wait on the host.
class Sam3GpuPipeline {
public:
    explicit Sam3GpuPipeline(Sam3PipelineConfig config);
    ~Sam3GpuPipeline();
    Sam3GpuPipeline(const Sam3GpuPipeline&) = delete;
    Sam3GpuPipeline& operator=(const Sam3GpuPipeline&) = delete;
    void Configure(const TextPrompt& prompt, bool temporal, size_t frame_count);
    void Reset();
    void Submit(const vision::DeviceRgbaView& source, void* host_stream, int64_t frame);
    bool HasFrame(int64_t frame) const;
    // selected_id=0: combined; otherwise 1..16. labels produces a colored ID preview.
    void Render(int64_t frame, const vision::DeviceRgbaView& source,
                const vision::DeviceOutputView& output, void* host_stream,
                int selected_id = 0, bool labels = false);
    void Synchronize();
    // Explicit offline boundary: waits for completion and downloads only logits.
    SavedMask Export(int64_t frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
