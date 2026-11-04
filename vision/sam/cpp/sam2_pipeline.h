// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "ort_session.h"
#include "sam2_common.h"
#include "sam2_cuda.h"
#include <onnxruntime_cxx_api.h>

namespace din::sam2
{

namespace ort
{
using din::common::BindOutput;
using din::common::CreateTensorRTRTXComputeStream;
using din::common::EpContextOptions;
using din::common::ModelProfile;
using din::common::NotificationPtr;
using din::common::OrtRunner;
using din::common::RegisterTensorRTRTXProvider;
using din::common::TensorBuffer;

using Fp16 = Ort::Float16_t;

template <typename T>
Ort::Value AliasTensorBytes(TensorBuffer<T>& buffer, OrtRunner& runner, bool use_device_io, size_t byte_offset,
                            size_t byte_count, const std::vector<int64_t>& shape,
                            ONNXTensorElementDataType element_type)
{
    if (use_device_io)
    {
        auto* base = static_cast<uint8_t*>(buffer.BindingValue().GetTensorMutableRawData());
        return Ort::Value::CreateTensor(runner.DeviceMemory(), base + byte_offset, byte_count, shape.data(),
                                        shape.size(), element_type);
    }
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto* base = reinterpret_cast<uint8_t*>(buffer.HostData());
    return Ort::Value::CreateTensor(memory, base + byte_offset, byte_count, shape.data(), shape.size(), element_type);
}
}  // namespace ort

namespace detail
{
class AsyncMaskWriter;
struct StreamHandoff;
struct TemporalCache;
}

struct Sam2PipelineConfig
{
    std::filesystem::path model_dir = "D:/models/sam2.1_base_plus_onnx";
    std::string provider = "trt-rtx";
    std::filesystem::path ep_cache_dir = "artifacts/sam2/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/sam2/ep_context";
    bool propagate = false;
    bool dump_masks = false;
};

class Sam2Pipeline
{
public:
    explicit Sam2Pipeline(Sam2PipelineConfig config);
    ~Sam2Pipeline();

    Image SegmentImage(const Image& image, const Prompt& prompt);

    // With a non-null host stream, enqueues input/output event dependencies and
    // returns before completion. Borrowed images must obey that stream's lifetime
    // contract; downstream work must use it or wait on it. Null stream completes
    // before return. For tracking call ResetSequence first; prompt is null.
    // A null output advances tracking without producing a display image.
    void ProcessDeviceFrame(const DeviceRgbaView& source, const DeviceOutputView* output,
                            void* host_stream, const Prompt* image_prompt = nullptr);

    // Bounded, device-resident logits and recurrent-state cache, owned by this
    // pipeline. Calls and resource reuse are ordered on its compute stream.
    void ClearTemporalCache();
    void RememberTemporalFrame(int64_t frame, bool checkpoint);
    bool HasTemporalMask(int64_t frame) const;
    bool RestoreTemporalCheckpoint(int64_t before, int64_t& restored);
    void RenderTemporalMask(int64_t frame, const DeviceRgbaView& source,
                            const DeviceOutputView& output, void* host_stream);
    Image ReadTemporalMask(int64_t frame, const Image& dimensions);

    // Calls on one pipeline must be serialized. Reset before starting a new clip.
    void ResetSequence(const Prompt& prompt);
    Image ProcessNextFrame(const Image& image, bool produce_mask = true);

    void RunBatch(const std::vector<std::filesystem::path>& frames, const Prompt& prompt,
                  const std::filesystem::path& output_path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace din::sam2
