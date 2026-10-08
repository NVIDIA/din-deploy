// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace din::image_gen
{

enum class PixelDiTProcessingBackend
{
    Cpu,
    Cuda,
};

enum class PixelDiTExecutionProvider
{
    TrtRtx,
};

struct PixelDiTConfig
{
    PixelDiTProcessingBackend processing = PixelDiTProcessingBackend::Cpu;
    PixelDiTExecutionProvider provider = PixelDiTExecutionProvider::TrtRtx;
    std::filesystem::path model_dir;
    std::filesystem::path ep_cache_dir = "artifacts/pixeldit/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/pixeldit/ep_context";
    std::filesystem::path output_path;
    std::string prompt;
    // Unset sampler options fall back to pipeline_config.json, then to built-in defaults.
    std::optional<std::string> negative_prompt;
    std::optional<int> steps;
    std::optional<float> cfg_scale;
    std::optional<float> flow_shift;
    unsigned int seed = 42;
    unsigned int num_images = 1;
    // Debug / validation: raw fp32 [1, 3, H, W] initial noise instead of the seeded generator, and a
    // directory for raw dumps of intermediate tensors.
    std::filesystem::path init_noise_path;
    std::filesystem::path dump_dir;
};

struct PixelDiTImage
{
    std::vector<float> data;  // [3, H, W] in [-1, 1]
    int height = 0;
    int width = 0;
};

class PixelDiTProcessingPipeline
{
public:
    virtual ~PixelDiTProcessingPipeline() = default;
    virtual void Initialize() = 0;
    virtual void SetPrompt(std::string prompt) = 0;
    virtual PixelDiTImage GenerateImage(unsigned int seed) = 0;
};

std::unique_ptr<PixelDiTProcessingPipeline> CreatePixelDiTPipeline(const PixelDiTConfig& config);

}  // namespace din::image_gen
