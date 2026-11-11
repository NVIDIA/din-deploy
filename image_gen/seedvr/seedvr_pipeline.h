// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "rng.h"
#include <onnxruntime_cxx_api.h>

namespace din::seedvr
{
struct ModelConfig
{
    std::filesystem::path model_dir = "models/SeedVR2-3B-onnx";
    std::string transformer_precision = "bf16";  // DiT compute; VAE and boundary tensors remain FP16.
    std::string vae_backend = "onnx";
    std::filesystem::path vae_weights_dir;
    bool vae_autotune = false;
    int vae_tuning_candidates = 5;
    int vae_workspace_mib = 256;
    std::filesystem::path vae_tuning_cache = "out/seedvr/cudnn_tuning_cache";
    std::filesystem::path ep_cache_dir = "out/seedvr/trt_rtx_cache_cpp";
    std::filesystem::path ep_context_dir = "out/seedvr/ep_context_cpp";
    // Empty disables streaming at construction; nonempty enables budget updates.
    std::string weight_streaming_budget;
    int vae_encoder_tile_size = 0;
    int vae_encoder_overlap = 64;
    int vae_decoder_tile_size = 0;
    int vae_decoder_overlap = 64;
};
struct ImageShape
{
    size_t source_height = 0, source_width = 0;
    size_t target_height = 0, target_width = 0;
};
struct VideoShape
{
    ImageShape image;
    size_t frames = 5;  // Exactly 4n+1, n >= 1.
};
struct VideoFrameRange
{
    size_t min;
    size_t max;
};
// Exporter dependency: export_seedvr.py captures dynamic video T=4n+1 in [5,33].
// Keep these bounds synchronized with its VIDEO_FRAME_RANGE; no sidecar is used.
inline constexpr VideoFrameRange kVideoFrameRange{5, 33};
struct DynamicOptions
{
    uint32_t seed = 0;
    common::Rng rng = common::Rng::CpuStd;
    float latent_noise_scale = 0.0f;
    std::string color_correction = "lab";
};
struct DynamicOptionsUpdate
{
    std::optional<uint32_t> seed;
    std::optional<common::Rng> rng;
    std::optional<float> latent_noise_scale;
    std::optional<std::string> color_correction;
    std::optional<std::string> weight_streaming_budget;
};
enum class ImageRole
{
    Input,
    Output
};
enum class ImageDevice
{
    Cpu,
    Cuda
};
namespace detail
{
class PipelineCore;
}

// RGB float32 contiguous CHW in [0,1]. Runs are synchronous and not thread-safe.
// Construction selects the ORT device's CUDA device; keep it current on the
// calling thread until the pipeline and its factory tensors are destroyed.
// CUDA inputs must be ready on entry. Factory tensors must die before the
// pipeline. Runs fill output storage without replacing it.
class ImagePipeline
{
public:
    ImagePipeline(ModelConfig, ImageShape, DynamicOptions = {});
    ~ImagePipeline();
    ImagePipeline(const ImagePipeline&) = delete;
    ImagePipeline& operator=(const ImagePipeline&) = delete;
    Ort::Value CreateImage(ImageRole, ImageDevice = ImageDevice::Cpu);
    void RunImage(const Ort::Value& input, Ort::Value& output);
    void UpdateOptions(const DynamicOptionsUpdate&);

private:
    std::unique_ptr<detail::PipelineCore> core_;
};
class VideoPipeline
{
public:
    VideoPipeline(ModelConfig, VideoShape, DynamicOptions = {});
    ~VideoPipeline();
    VideoPipeline(const VideoPipeline&) = delete;
    VideoPipeline& operator=(const VideoPipeline&) = delete;
    Ort::Value CreateImage(ImageRole, ImageDevice = ImageDevice::Cpu);
    // Exactly T inputs and outputs. No temporal padding.
    void RunChunk(std::span<const Ort::Value> inputs, std::span<Ort::Value> outputs);
    void UpdateOptions(const DynamicOptionsUpdate&);

private:
    std::unique_ptr<detail::PipelineCore> core_;
};
}  // namespace din::seedvr
