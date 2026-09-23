// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "flux2_cli.h"
#include "tokenizer.h"

// ============================================================================
// Configuration
// ============================================================================

#ifdef _WIN32
static const std::filesystem::path DEFAULT_MODEL_BASE_PATH = "S:/din_deploy_artifacts/FLUX.2-klein-4B-onnx";
#else
static const std::filesystem::path DEFAULT_MODEL_BASE_PATH = "/mnt/share/onnx/FLUX.2-klein-4B-onnx/";
#endif

struct Flux2ModelPaths
{
    std::filesystem::path base_dir;
    std::filesystem::path text_encoder_model;
    std::filesystem::path transformer_model;
    std::filesystem::path vae_decoder_model;
    std::filesystem::path tokenizer_dir;
};

struct Flux2ModelCachePaths
{
    std::string text_encoder;
    std::string transformer;
    std::string vae_decoder;
};

inline Flux2ModelPaths MakeFlux2ModelPaths(std::filesystem::path model_dir, const std::string& precision,
                                           Flux2TextEncoder encoder = Flux2TextEncoder::Qwen3_4B)
{
    if (model_dir.empty())
    {
        model_dir = DEFAULT_MODEL_BASE_PATH;
    }
    return {
        model_dir,
        model_dir / (encoder == Flux2TextEncoder::Qwen3_4B ? "text_encoder" : "text_encoder_translator") / "model.onnx",
        model_dir / ("transformer_" + precision) / "model.onnx",
        model_dir / "vae_decoder/model.onnx",
        model_dir / "tokenizer",
    };
}

inline Flux2ModelCachePaths MakeFlux2ModelCachePaths(const std::string& precision, const std::string& prefix = {},
                                                     Flux2TextEncoder encoder = Flux2TextEncoder::Qwen3_4B,
                                                     bool weight_streaming = false)
{
    const std::string prefix_separator = prefix.empty() ? "" : prefix + "_";
    return {
        prefix_separator + (encoder == Flux2TextEncoder::Qwen3_4B ? "text_encoder" : "text_encoder_translator"),
        prefix_separator + "transformer_" + precision + (weight_streaming ? "_ws" : ""),
        prefix_separator + "vae_decoder",
    };
}

static const std::filesystem::path TEXT_ENCODER_MODEL_CTX = "text_encoder_model_ctx.onnx";
static const std::filesystem::path TRANSFORMER_MODEL_CTX = "transformer_model_ctx.onnx";
static const std::filesystem::path VAE_DECODER_MODEL_CTX = "vae_decoder_model_ctx.onnx";

static constexpr int64_t BATCH_SIZE = 1;
static constexpr int64_t SEQUENCE_LENGTH = 512;
static constexpr int64_t TEXT_ENCODER_EMBED_DIM = 7680;

static constexpr int64_t IMAGE_HEIGHT = 1024;
static constexpr int64_t IMAGE_WIDTH = 1024;
static constexpr int64_t IMAGE_CHANNELS = 3;

static constexpr int64_t LATENT_HEIGHT = IMAGE_HEIGHT / 16;  // 64
static constexpr int64_t LATENT_WIDTH = IMAGE_WIDTH / 16;    // 64
static constexpr int64_t LATENT_CHANNELS = 128;
static constexpr int64_t PATCH_SIZE = 2;

static constexpr int64_t IMAGE_SEQUENCE = LATENT_HEIGHT * LATENT_WIDTH;
static constexpr int64_t TRANSFORMER_HIDDEN_DIM = IMAGE_SEQUENCE;

static constexpr int FLOW_STEPS = 4;
static constexpr float T_START = 1.0f;
static constexpr float T_END = 0.0f;

inline constexpr const char* DEFAULT_PROMPT =
    "a photo of a forest with mist swirling around the tree trunks. "
    "The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture";

// ============================================================================
// Product of shape dimensions
// ============================================================================

inline size_t shape_numel(const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto v : shape)
        n *= v;
    return n;
}

struct Flux2TextEncoderInputs
{
    std::vector<int64_t> token_ids;
    int64_t pad_token_id = 0;
};

inline std::string FormatFlux2Prompt(const std::string& prompt)
{
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

inline std::vector<float> MakeFlux2Schedule(int steps)
{
    if (steps < 1 || steps > 50)
        throw std::invalid_argument("Denoise steps must be in [1, 50]");
    constexpr double m200 = 0.00016927 * IMAGE_SEQUENCE + 0.45666666;
    constexpr double m10 = 8.73809524e-05 * IMAGE_SEQUENCE + 1.89833333;
    constexpr double a = (m200 - m10) / 190.0;
    const double mu = IMAGE_SEQUENCE > 4300 ? m200 : a * steps + m200 - 200.0 * a;
    const double e = std::exp(mu);
    std::vector<float> schedule(steps + 1);
    for (int i = 0; i < steps; ++i)
    {
        const double t = 1.0 - static_cast<double>(i) / steps;
        schedule[i] = static_cast<float>(e / (e + 1.0 / t - 1.0));
    }
    schedule.back() = 0.0f;
    return schedule;
}

inline Flux2TextEncoderInputs TokenizeFlux2Prompt(const Flux2ModelPaths& model_paths, const std::string& prompt)
{
    din::io::Tokenizer tokenizer((model_paths.tokenizer_dir / "tokenizer.json").string(),
                                 din::io::TokenizerFormat::Json);

    const std::string chat_text = FormatFlux2Prompt(prompt);
    Flux2TextEncoderInputs inputs;
    inputs.token_ids = tokenizer.Encode(chat_text, false);
    inputs.pad_token_id = tokenizer.TokenId("<|endoftext|>");
    if (inputs.pad_token_id < 0)
    {
        throw std::runtime_error("Flux2 tokenizer is missing <|endoftext|> pad token");
    }
    return inputs;
}

inline void FillTextEncoderInputs(const std::vector<int64_t>& token_ids, int64_t pad_token_id, int64_t* tokens_data,
                                  int64_t* attention_mask_data, int64_t batch_size, int64_t sequence_length)
{
    const size_t prompt_length = std::min(token_ids.size(), static_cast<size_t>(sequence_length));
    if (token_ids.size() > static_cast<size_t>(sequence_length))
    {
        std::cout << "Prompt tokens truncated from " << token_ids.size() << " to " << sequence_length << std::endl;
    }

    for (int64_t b = 0; b < batch_size; ++b)
    {
        for (int64_t t = 0; t < sequence_length; ++t)
        {
            const size_t token_index = static_cast<size_t>(t);
            const int64_t output_index = b * sequence_length + t;
            tokens_data[output_index] = token_index < prompt_length ? token_ids[token_index] : pad_token_id;
            attention_mask_data[output_index] = token_index < prompt_length ? 1 : 0;
        }
    }
}

// ============================================================================
// Random latent initialisation
// ============================================================================

inline void initialize_latent(float* latent, size_t num_elements, unsigned int seed = 42)
{
    std::default_random_engine rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i)
        latent[i] = dist(rng);
}

// ============================================================================
// Latent post-processing
// ============================================================================

inline void unpack_latents_with_ids(const float* in,  // [H*W, C]  tokens in raster order
                                    float* out,       // [C, H, W]
                                    int H, int W, int C)
{
    int HW = H * W;
    for (int i = 0; i < HW; ++i)
        for (int c = 0; c < C; ++c)
            out[c * HW + i] = in[i * C + c];
}

inline void unpatchify_latents(const float* in,  // [B, C*pi*pj, I, J]
                               float* out,       // [B, C, I*pi, J*pj]
                               int B, int C, int I, int J, int pi, int pj)
{
    const int C_in = C * pi * pj;
    const int I_out = I * pi;
    const int J_out = J * pj;

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < I; ++i)
                for (int j = 0; j < J; ++j)
                    for (int p = 0; p < pi; ++p)
                        for (int q = 0; q < pj; ++q)
                        {
                            int in_c = c * pi * pj + p * pj + q;
                            int in_idx = ((b * C_in + in_c) * I + i) * J + j;
                            int out_idx = ((b * C + c) * I_out + (i * pi + p)) * J_out + (j * pj + q);
                            out[out_idx] = in[in_idx];
                        }
}

inline void denormalize_latents(float* data,           // [C, H*W]  contiguous (C, H, W)
                                const float* bn_mean,  // [C]
                                const float* bn_std,   // [C]  already sqrt(var + eps)
                                int C, int HW)
{
    for (int c = 0; c < C; ++c)
    {
        float m = bn_mean[c];
        float s = bn_std[c];
        float* ch = data + c * HW;
        for (int i = 0; i < HW; ++i)
            ch[i] = ch[i] * s + m;
    }
}

// ============================================================================
// Debug stats
// ============================================================================

inline void stats(const float* data, size_t count)
{
    if (count == 0)
        return;
    float mn = *std::min_element(data, data + count);
    float mx = *std::max_element(data, data + count);
    float sum = std::accumulate(data, data + count, 0.0f);
    float mean = sum / count;
    float accum = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        float d = data[i] - mean;
        accum += d * d;
    }
    float stdev = std::sqrt(accum / (count > 1 ? count - 1 : 1));
    std::printf("min=%.4f  max=%.4f  mean=%.4f  stddev=%.4f\n", mn, mx, mean, stdev);
}

inline void stats(const std::vector<float>& vec)
{
    stats(vec.data(), vec.size());
}
