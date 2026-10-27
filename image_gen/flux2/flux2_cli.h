// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rng.h"

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

inline void ValidateFlux2Rng(din::common::Rng rng, Flux2ProcessingBackend processing)
{
    using din::common::Rng;
    if (rng != Rng::CpuStd && rng != Rng::CudaPhilox && rng != Rng::CpuPhilox)
        throw std::invalid_argument("Unknown random number generator");
    if (rng == Rng::CudaPhilox && processing != Flux2ProcessingBackend::Cuda)
        throw std::invalid_argument("cuda-philox RNG requires --processing cuda");
}

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

struct Flux2ModelConfig
{
    std::filesystem::path model_path;
    std::filesystem::path ep_cache_dir = "artifacts/flux2/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/flux2/ep_context";
    std::string cache_subpath;
    bool operator==(const Flux2ModelConfig&) const = default;
};

struct Flux2TextEncoderConfig : Flux2ModelConfig
{
    Flux2TextEncoder variant = Flux2TextEncoder::Qwen3_4B;
    std::filesystem::path tokenizer_dir;
    bool operator==(const Flux2TextEncoderConfig&) const = default;
};

struct Flux2TransformerConfig : Flux2ModelConfig
{
    std::string precision = "bf16";
    // Empty keeps a non-streaming engine; -1 enables automatic streaming.
    std::string initial_weight_streaming_budget;
    bool operator==(const Flux2TransformerConfig&) const = default;
};

struct Flux2VaeConfig : Flux2ModelConfig
{
    std::filesystem::path latent_stats_path;
    bool operator==(const Flux2VaeConfig&) const = default;
};

// Recoverable: the affected stage is unavailable; other sessions remain loaded.
class Flux2StageLoadError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct Flux2Config
{
    Flux2ProcessingBackend processing = Flux2ProcessingBackend::Cuda;
    Flux2ExecutionProvider provider = Flux2ExecutionProvider::TrtRtx;
    Flux2TextEncoderConfig text_encoder;
    Flux2TransformerConfig transformer;
    Flux2VaeConfig vae;
    int steps = 4;
    std::filesystem::path output_path;
    std::string prompt;
    unsigned int seed = 42;
    // CPU standard noise, CUDA Philox, or CPU Philox with fixed H100 SXM5 ordering.
    din::common::Rng rng = din::common::Rng::CpuStd;
    unsigned int num_images = 5;
};

struct Flux2GenerationOptions
{
    // Omit to restore the startup budget. Applied via transformer session dynamic options only.
    std::optional<std::string> weight_streaming_budget;
    // Omit to use the startup selections. Overrides apply to this generation only.
    std::optional<int> steps;
    std::optional<din::common::Rng> rng;
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
    virtual void UpdateModels(const Flux2Config& config) = 0;
    virtual void SetPrompt(std::string prompt) = 0;
    // Opaque identity for diagnostics; never dereference or retain after pipeline destruction.
    virtual const void* TransformerSessionIdentity() const = 0;
    virtual const void* TextEncoderSessionIdentity() const = 0;
    virtual const void* VaeSessionIdentity() const = 0;
    virtual Flux2Image GenerateImage(unsigned int seed, const Flux2Progress& progress = {},
                                     const Flux2GenerationOptions& options = {}) = 0;
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2Pipeline(const Flux2Config& config);

void ValidateFlux2Config(const Flux2Config& config);
// Resolve existing CLI/UI root controls into independent stage settings.
void ConfigureFlux2Models(Flux2Config& config, const std::filesystem::path& model_root,
                          const std::filesystem::path& cache_root, const std::filesystem::path& context_root);
bool IsFlux2BackendAvailable(Flux2ProcessingBackend backend);
