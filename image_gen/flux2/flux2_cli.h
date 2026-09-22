// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
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

enum class Flux2TextEncoder
{
    Qwen3_4B,
    Qwen3_06BTranslator
};

// Called on the generation thread. Completed denoising steps are in [0, total].
using Flux2Progress = std::function<void(const char* stage, int completed, int total)>;

struct Flux2Timings
{
    double encode_ms = 0;
    double rng_ms = 0;
    double denoise_ms = 0;
    double decode_ms = 0;
    double total_ms = 0;
    double per_step_ms = 0;
};

struct Flux2Config
{
    Flux2ProcessingBackend processing = Flux2ProcessingBackend::Cuda;
    Flux2ExecutionProvider provider = Flux2ExecutionProvider::TrtRtx;
    std::filesystem::path model_dir;
    std::string precision = "bf16";
    Flux2TextEncoder text_encoder = Flux2TextEncoder::Qwen3_4B;
    int steps = 4;
    // Startup budget: empty keeps a non-streaming engine; -1 enables automatic streaming.
    std::string weight_streaming_budget;
    std::filesystem::path ep_cache_dir = "artifacts/flux2/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/flux2/ep_context";
    std::filesystem::path output_path;
    std::string prompt;
    unsigned int seed = 42;
    unsigned int num_images = 5;
};

struct Flux2GenerationOptions
{
    // Omit to restore the startup budget. Applied via transformer session dynamic options only.
    std::optional<std::string> weight_streaming_budget;
};

struct Flux2Image
{
    std::vector<float> data;
    int height = 0;
    int width = 0;
    Flux2Timings timings;
};

class Flux2ProcessingPipeline
{
public:
    virtual ~Flux2ProcessingPipeline() = default;
    virtual void Initialize() = 0;
    virtual void SetPrompt(std::string prompt) = 0;
    // Opaque identity for diagnostics; never dereference or retain after pipeline destruction.
    virtual const void* TransformerSessionIdentity() const = 0;
    virtual Flux2Image GenerateImage(unsigned int seed, const Flux2Progress& progress = {},
                                     const Flux2GenerationOptions& options = {}) = 0;
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2Pipeline(const Flux2Config& config);

void ValidateFlux2Config(const Flux2Config& config);
bool IsFlux2BackendAvailable(Flux2ProcessingBackend backend);
