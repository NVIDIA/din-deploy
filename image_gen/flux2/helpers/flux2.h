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

inline Flux2ModelPaths MakeFlux2ModelPaths(std::filesystem::path model_dir)
{
    if (model_dir.empty())
    {
        model_dir = DEFAULT_MODEL_BASE_PATH;
    }
    return {
        model_dir,
        model_dir / "text_encoder/model.onnx",
        model_dir / "transformer/model.onnx",
        model_dir / "vae_decoder/model.onnx",
        model_dir / "tokenizer",
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

inline Flux2TextEncoderInputs TokenizeFlux2Prompt(const Flux2ModelPaths& model_paths, const std::string& prompt)
{
    din::io::Tokenizer tokenizer((model_paths.tokenizer_dir / "tokenizer.json").string(),
                                 din::io::TokenizerFormat::Json);

    const std::string chat_text = "<|im_start|>user\n" + prompt + "<|im_end|>";
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

// ============================================================================
// Batch-norm parameters (exported from vae.bn)
// ============================================================================

static const float BN_STD[128] = {
    1.8046875, 1.7734375, 1.7890625, 1.7890625, 1.7734375, 1.7578125, 1.7578125, 1.75,      1.734375,  1.734375,
    1.734375,  1.734375,  1.859375,  1.8515625, 1.859375,  1.859375,  1.7578125, 1.7578125, 1.7578125, 1.7578125,
    1.734375,  1.7421875, 1.734375,  1.7421875, 1.734375,  1.7265625, 1.7421875, 1.734375,  1.7578125, 1.75,
    1.7578125, 1.75,      1.84375,   1.84375,   1.8515625, 1.8515625, 1.828125,  1.78125,   1.7890625, 1.796875,
    1.8046875, 1.7890625, 1.765625,  1.765625,  1.7734375, 1.765625,  1.796875,  1.7890625, 1.7578125, 1.7421875,
    1.75,      1.734375,  1.796875,  1.796875,  1.796875,  1.796875,  1.7734375, 1.7734375, 1.7734375, 1.7734375,
    1.7578125, 1.7421875, 1.765625,  1.75,      1.7734375, 1.7734375, 1.78125,   1.78125,   1.8125,    1.8046875,
    1.8046875, 1.8046875, 1.7578125, 1.734375,  1.7578125, 1.75,      1.7578125, 1.75,      1.765625,  1.765625,
    1.765625,  1.7578125, 1.78125,   1.7734375, 1.7890625, 1.796875,  1.7890625, 1.7734375, 1.765625,  1.765625,
    1.7578125, 1.7578125, 1.7734375, 1.765625,  1.78125,   1.78125,   1.71875,   1.703125,  1.7265625, 1.7109375,
    1.78125,   1.7578125, 1.75,      1.7421875, 1.7578125, 1.75,      1.7578125, 1.734375,  1.765625,  1.7578125,
    1.7421875, 1.7421875, 1.75,      1.7578125, 1.765625,  1.765625,  1.7265625, 1.7109375, 1.734375,  1.7265625,
    1.765625,  1.765625,  1.7734375, 1.765625,  1.7421875, 1.7421875, 1.7578125, 1.7578125};

static const float BN_MEAN[128] = {
    -0.0673828125f,         -0.0712890625f,         -0.0751953125f,        -0.07470703125f,        0.0223388671875f,
    0.0179443359375f,       0.01422119140625f,      0.018310546875f,       -6.29425048828125e-05f, -0.006256103515625f,
    -0.000209808349609375f, -0.003143310546875f,    -0.0272216796875f,     -0.028076171875f,       -0.027587890625f,
    -0.029052734375f,       -0.07666015625f,        -0.0673828125f,        -0.09033203125f,        -0.08935546875f,
    0.016845703125f,        0.01519775390625f,      0.00787353515625f,     0.00860595703125f,      0.00836181640625f,
    0.0015411376953125f,    0.0002574920654296875f, -0.0042724609375f,     -0.0439453125f,         -0.0419921875f,
    -0.043701171875f,       -0.043212890625f,       -0.01025390625f,       -0.01318359375f,        -0.006622314453125f,
    -0.0047607421875f,      -0.031005859375f,       -0.030517578125f,      -0.0279541015625f,      -0.0179443359375f,
    0.003021240234375f,     0.00150299072265625f,   0.0125732421875f,      0.01446533203125f,      0.03466796875f,
    0.03369140625f,         0.03369140625f,         0.0283203125f,         0.001983642578125f,     0.004730224609375f,
    0.004669189453125f,     0.004974365234375f,     0.01226806640625f,     0.00811767578125f,      0.008056640625f,
    0.01458740234375f,      0.06787109375f,         0.06787109375f,        0.07666015625f,         0.0732421875f,
    -0.046142578125f,       -0.04736328125f,        -0.039306640625f,      -0.051025390625f,       -0.052734375f,
    -0.0478515625f,         -0.047119140625f,       -0.0517578125f,        -0.03173828125f,        -0.03173828125f,
    -0.034423828125f,       -0.0281982421875f,      0.051025390625f,       0.04443359375f,         0.057861328125f,
    0.0458984375f,          -0.041259765625f,       -0.0458984375f,        -0.048828125f,          -0.046630859375f,
    -0.00885009765625f,     -0.0106201171875f,      -0.0087890625f,        -0.004608154296875f,    -0.03759765625f,
    -0.043212890625f,       -0.04345703125f,        -0.0498046875f,        0.0118408203125f,       0.0166015625f,
    0.020263671875f,        0.0279541015625f,       0.01129150390625f,     0.01287841796875f,      0.001556396484375f,
    0.00714111328125f,      -0.01177978515625f,     -0.00183868408203125f, -0.01416015625f,        -0.00537109375f,
    -0.00909423828125f,     -0.0137939453125f,      -0.01446533203125f,    -0.0186767578125f,      0.0322265625f,
    0.030517578125f,        0.02587890625f,         0.0299072265625f,      0.053955078125f,        0.0615234375f,
    0.049560546875f,        0.05908203125f,         -0.051025390625f,      -0.060302734375f,       -0.0478515625f,
    -0.052490234375f,       -0.022705078125f,       -0.0274658203125f,     -0.015380859375f,       -0.0255126953125f,
    -0.05712890625f,        -0.056396484375f,       -0.0517578125f,        -0.049560546875f,       0.0115966796875f,
    0.00543212890625f,      0.016357421875f,        0.0103759765625f};
