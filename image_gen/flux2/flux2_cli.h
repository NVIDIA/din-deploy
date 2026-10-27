// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

enum class Flux2ProcessingBackend
{
    Cpu,
    Cuda,
    Dx,
    DxCig,
    Vk,
    VkCig,
};

enum class Flux2ExecutionProvider
{
    Cpu,
    TrtRtx,
};

struct Flux2Config
{
    Flux2ProcessingBackend processing = Flux2ProcessingBackend::Cuda;
    Flux2ExecutionProvider provider = Flux2ExecutionProvider::TrtRtx;
    std::filesystem::path model_dir;
    std::string precision = "bf16";
    std::filesystem::path ep_cache_dir = "artifacts/flux2/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/flux2/ep_context";
    std::filesystem::path output_path;
    std::string prompt;
    unsigned int seed = 42;
    unsigned int num_images = 5;
};

struct Flux2Image
{
    std::vector<float> data;
    int height = 0;
    int width = 0;
};

class Flux2ProcessingPipeline
{
public:
    virtual ~Flux2ProcessingPipeline() = default;
    virtual void Initialize() = 0;
    virtual void SetPrompt(std::string prompt) = 0;
    virtual Flux2Image GenerateImage(unsigned int seed) = 0;
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2Pipeline(const Flux2Config& config);
