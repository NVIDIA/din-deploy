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
}

struct Sam2PipelineConfig
{
    std::filesystem::path model_dir = "artifacts/sam2/onnx";
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

    Image SegmentImage(const Image& image, const Prompt& prompt);

    void RunBatch(const std::vector<std::filesystem::path>& frames, const Prompt& prompt,
                  const std::filesystem::path& output_path);

private:
    void AllocateBuffers();
    void PreparePromptBuffers(const Prompt& prompt);
    void ResizeImageRgbBufferIfNeeded(const Image& image);
    void ResizeMaskRgbBufferIfNeeded(const Image& image);
    void RunFrame(const Image& image, const std::filesystem::path& mask_path, bool write_mask,
                  detail::AsyncMaskWriter* writer);

    void RunEncoder(const Image& image);
    void RunFastMemoryAttention();
    void RunFastDecoder(bool prompted);
    void RunMaskPostprocess(const Image& image);
    void RunFastMemoryEncoder(bool prompted);

    Sam2PipelineConfig config_;
    Sam2Config model_config_;
    Ort::Env env_;
    Ort::SyncStream compute_stream_{nullptr};
    bool use_device_io_ = false;

    std::unique_ptr<ort::OrtRunner> image_preprocess_;
    std::unique_ptr<ort::OrtRunner> encoder_;
    std::unique_ptr<ort::OrtRunner> decoder_prompt_;
    std::unique_ptr<ort::OrtRunner> decoder_propagate_;
    std::unique_ptr<ort::OrtRunner> attention_;
    std::unique_ptr<ort::OrtRunner> memory_encoder_;
    std::unique_ptr<ort::OrtRunner> mask_postprocess_;

    std::unique_ptr<Ort::IoBinding> image_preprocess_binding_;
    std::unique_ptr<Ort::IoBinding> encoder_binding_;
    std::unique_ptr<Ort::IoBinding> decoder_prompt_binding_;
    std::unique_ptr<Ort::IoBinding> decoder_propagate_binding_;
    std::unique_ptr<Ort::IoBinding> memory_attention_binding_;
    std::unique_ptr<Ort::IoBinding> mask_postprocess_binding_;
    std::vector<Ort::Value> memory_feature_slots_;
    std::vector<Ort::Value> memory_pos_slots_;
    std::vector<std::unique_ptr<Ort::IoBinding>> memory_encoder_prompt_bindings_;
    std::vector<std::unique_ptr<Ort::IoBinding>> memory_encoder_propagate_bindings_;

    Ort::RunOptions run_options_;

    std::unique_ptr<ort::TensorBuffer<uint8_t>> image_rgb_;
    std::unique_ptr<ort::TensorBuffer<int64_t>> mask_output_size_;
    std::unique_ptr<ort::TensorBuffer<uint8_t>> mask_rgb_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> image_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> image_feature_0_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> image_feature_1_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> image_embeddings_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> vision_feat_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> vision_pos_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> prompt_coords_;
    std::unique_ptr<ort::TensorBuffer<int32_t>> prompt_labels_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> empty_coords_;
    std::unique_ptr<ort::TensorBuffer<int32_t>> empty_labels_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> input_masks_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> has_input_masks_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> mask_from_pts_true_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> mask_from_pts_false_;

    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> conditioned_embeddings_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> masks_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> low_res_masks_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> object_score_logits_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> memory_ring_;
    std::unique_ptr<ort::TensorBuffer<ort::Fp16>> memory_pos_ring_;

    size_t active_memory_ = 0;
    size_t next_slot_ = 0;
    uint32_t image_rgb_width_ = 0;
    uint32_t image_rgb_height_ = 0;
    uint32_t mask_rgb_width_ = 0;
    uint32_t mask_rgb_height_ = 0;
};

}  // namespace din::sam2
