// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tokenizer.h"

namespace din::image_gen
{

// ============================================================================
// Configuration
// ============================================================================

#ifdef _WIN32
static const std::filesystem::path DEFAULT_PIXELDIT_MODEL_DIR = "S:/din_deploy_artifacts/PixelDiT-1300M-1024px-onnx";
#else
static const std::filesystem::path DEFAULT_PIXELDIT_MODEL_DIR = "/mnt/share/onnx/PixelDiT-1300M-1024px-onnx/";
#endif

struct PixelDiTModelPaths
{
    std::filesystem::path base_dir;
    std::filesystem::path text_encoder_model;
    std::filesystem::path transformer_model;
    std::filesystem::path tokenizer_json;
    std::filesystem::path pipeline_config;
};

inline PixelDiTModelPaths MakePixelDiTModelPaths(std::filesystem::path model_dir)
{
    if (model_dir.empty())
    {
        model_dir = DEFAULT_PIXELDIT_MODEL_DIR;
    }
    return {
        model_dir,
        model_dir / "text_encoder/model.onnx",
        model_dir / "transformer/model.onnx",
        model_dir / "tokenizer/tokenizer.json",
        model_dir / "pipeline_config.json",
    };
}

inline constexpr const char* DEFAULT_PIXELDIT_PROMPT =
    "A red fox sitting in a snowy forest at sunrise, soft golden light, photorealistic";

// Upstream t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml text_encoder.chi_prompt, joined with "\n"
// (same as model_export/pixeldit/pipeline.py CHI_PROMPT).
inline constexpr const char* PIXELDIT_CHI_PROMPT =
    "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for "
    "image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial "
    "relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly without "
    "overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping "
    "peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street "
    "lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass "
    "skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:\n"
    "User Prompt: ";

inline constexpr const char* PIXELDIT_DEFAULT_NEGATIVE_PROMPT =
    "low quality, worst quality, over-saturated, blurry, deformed, watermark";

// pipeline_config.json written by export_pixeldit.py. Every field has a built-in default matching the
// 1024x1024 export; LoadPixelDiTPipelineConfig records which defaults were used.
struct PixelDiTPipelineConfig
{
    int64_t height = 1024;
    int64_t width = 1024;
    int64_t dit_batch = 2;  // [negative, positive] for classifier-free guidance
    float timestep_scale = 1000.0f;

    std::string chi_prompt = PIXELDIT_CHI_PROMPT;
    int64_t text_seq_len = 506;  // chi_prompt tokens (208) + txt_max_length (300) - 2
    int64_t txt_max_length = 300;
    int64_t bos_token_id = 2;
    int64_t pad_token_id = 0;
    std::vector<int64_t> positive_select_index;  // BOS + last (txt_max_length - 1) positions
    std::vector<int64_t> negative_select_index;  // first txt_max_length positions

    int steps = 50;
    float cfg_scale = 2.75f;
    float flow_shift = 4.0f;
    std::string negative_prompt = PIXELDIT_DEFAULT_NEGATIVE_PROMPT;
    float t_start = 1.0f;
    float t_end = 0.001f;
    int order = 2;

    bool file_found = false;
    std::vector<std::string> defaults_used;
};

PixelDiTPipelineConfig LoadPixelDiTPipelineConfig(const std::filesystem::path& path);

inline std::vector<int64_t> DefaultPositiveSelectIndex(int64_t text_seq_len, int64_t txt_max_length)
{
    std::vector<int64_t> index{0};
    for (int64_t i = text_seq_len - txt_max_length + 1; i < text_seq_len; ++i)
    {
        index.push_back(i);
    }
    return index;
}

inline std::vector<int64_t> DefaultNegativeSelectIndex(int64_t txt_max_length)
{
    std::vector<int64_t> index(static_cast<size_t>(txt_max_length));
    for (int64_t i = 0; i < txt_max_length; ++i)
    {
        index[static_cast<size_t>(i)] = i;
    }
    return index;
}

// ============================================================================
// Text encoder inputs
// ============================================================================

struct PixelDiTTextInputs
{
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    size_t num_tokens = 0;  // before padding / truncation
};

// Python str.strip() for ASCII whitespace.
inline std::string StripWhitespace(const std::string& text)
{
    const char* whitespace = " \t\n\r\f\v";
    const auto first = text.find_first_not_of(whitespace);
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = text.find_last_not_of(whitespace);
    return text.substr(first, last - first + 1);
}

// HF tokenizer(text, max_length=seq_len, padding="max_length", truncation=True): <bos> + tokens, truncated,
// right-padded with pad_token_id.
inline PixelDiTTextInputs MakeTextEncoderInputs(const din::io::Tokenizer& tokenizer, const std::string& text,
                                                int64_t seq_len, int64_t pad_token_id)
{
    const std::vector<int64_t> ids = tokenizer.Encode(text, true);
    PixelDiTTextInputs inputs;
    inputs.num_tokens = ids.size();
    inputs.input_ids.assign(static_cast<size_t>(seq_len), pad_token_id);
    inputs.attention_mask.assign(static_cast<size_t>(seq_len), 0);
    const size_t kept = std::min(ids.size(), static_cast<size_t>(seq_len));
    for (size_t i = 0; i < kept; ++i)
    {
        inputs.input_ids[i] = ids[i];
        inputs.attention_mask[i] = 1;
    }
    return inputs;
}

// ============================================================================
// Random noise (same generator as Flux2's initialize_latent)
// ============================================================================

inline void initialize_noise(float* noise, size_t num_elements, unsigned int seed)
{
    std::default_random_engine rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i)
        noise[i] = dist(rng);
}

// ============================================================================
// Flow DPM-Solver++ (2M) sampling math, host side.
//
// Mirrors model_export/pixeldit/pipeline.py (and upstream dpm_solver.py) operation by operation in fp32,
// so results match the Python reference up to libm differences in the scalar log/exp terms.
// Build with floating-point contraction disabled (see CMakeLists.txt).
// ============================================================================

// Runs fn(begin, end) over [0, n) on several threads. The sampling math is elementwise, so results are
// identical to a single-threaded loop.
template <typename Fn>
void ParallelFor(size_t n, Fn&& fn)
{
    constexpr size_t min_chunk = size_t{1} << 16;
    const size_t threads =
        std::min<size_t>({std::max(1u, std::thread::hardware_concurrency()), size_t{16}, (n + min_chunk - 1) / min_chunk});
    if (threads <= 1)
    {
        fn(size_t{0}, n);
        return;
    }
    const size_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (size_t k = 1; k < threads; ++k)
    {
        const size_t begin = k * chunk;
        const size_t end = std::min(n, begin + chunk);
        workers.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    fn(size_t{0}, std::min(n, chunk));
    for (auto& worker : workers)
    {
        worker.join();
    }
}

// DPM_Solver.get_time_steps("time_uniform_flow"): torch.linspace(t_start, t_end, steps + 1) in fp32, then
// t = flip(shift * s / (1 + (shift - 1) * s)) with s = 1 - linspace. Returns steps + 1 values, ~1 down to 0.
inline std::vector<float> flow_timesteps(int steps, float flow_shift, float t_start, float t_end)
{
    const int count = steps + 1;
    const float step = (t_end - t_start) / static_cast<float>(count - 1);
    const int halfway = count / 2;
    std::vector<float> betas(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        betas[static_cast<size_t>(i)] =
            i < halfway ? t_start + step * static_cast<float>(i) : t_end - step * static_cast<float>(count - i - 1);
    }
    std::vector<float> timesteps(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const float s = 1.0f - betas[static_cast<size_t>(i)];
        const float numerator = flow_shift * s;
        const float scaled = (flow_shift - 1.0f) * s;
        const float denominator = 1.0f + scaled;
        timesteps[static_cast<size_t>(count - 1 - i)] = numerator / denominator;
    }
    return timesteps;
}

// Noise schedule (NoiseScheduleFlow): alpha_t = 1 - t, sigma_t = t, lambda_t = log(alpha_t) - log(sigma_t).
inline float flow_log_alpha(float t)
{
    return std::log(1.0f - t);
}

inline float flow_lambda(float t)
{
    return flow_log_alpha(t) - std::log(t);
}

// velocity: [2, n] DiT output for [negative, positive]; x: [n] current sample at time t.
// noise_b = (1 - t) * v_b + x;  noise = noise_u + cfg * (noise_c - noise_u);  x0 = (x - t * noise) / (1 - t).
inline void cfg_data_prediction(const float* velocity, const float* x, float t, float cfg_scale, float* x0, size_t n)
{
    const float one_minus_t = 1.0f - t;
    const float* v_uncond = velocity;
    const float* v_cond = velocity + n;
    ParallelFor(n,
                [=](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const float scaled_u = one_minus_t * v_uncond[i];
                        const float noise_u = scaled_u + x[i];
                        const float scaled_c = one_minus_t * v_cond[i];
                        const float noise_c = scaled_c + x[i];
                        const float diff = noise_c - noise_u;
                        const float guided_delta = cfg_scale * diff;
                        const float noise = noise_u + guided_delta;
                        const float t_noise = t * noise;
                        const float numerator = x[i] - t_noise;
                        x0[i] = numerator / one_minus_t;
                    }
                });
}

// DPM-Solver-1 (dpmsolver++) from s to t: x_t = (t / s) * x - (alpha_t * phi_1) * x0_s, phi_1 = expm1(-h).
// At t == 0 (sigma_t = 0, alpha_t = 1, expm1(-inf) = -1) the update is exactly x0_s.
inline void dpm_first_order_update(float* x, const float* model_s, float s, float t, size_t n)
{
    if (t == 0.0f)
    {
        std::copy_n(model_s, n, x);
        return;
    }
    const float h = flow_lambda(t) - flow_lambda(s);
    const float alpha_t = std::exp(flow_log_alpha(t));
    const float phi_1 = std::expm1(-h);
    const float ratio = t / s;
    const float coeff = alpha_t * phi_1;
    ParallelFor(n,
                [=](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const float a = ratio * x[i];
                        const float b = coeff * model_s[i];
                        x[i] = a - b;
                    }
                });
}

// Multistep DPM-Solver-2 (dpmsolver++): uses x0 at t_prev_1 and t_prev_0.
inline void dpm_second_order_update(float* x, const float* model_prev_1, const float* model_prev_0, float t_prev_1,
                                    float t_prev_0, float t, size_t n)
{
    const float lambda_prev_1 = flow_lambda(t_prev_1);
    const float lambda_prev_0 = flow_lambda(t_prev_0);
    const float lambda_t = flow_lambda(t);
    const float alpha_t = std::exp(flow_log_alpha(t));
    const float h_0 = lambda_prev_0 - lambda_prev_1;
    const float h = lambda_t - lambda_prev_0;
    const float r0 = h_0 / h;
    const float inv_r0 = 1.0f / r0;
    const float phi_1 = std::expm1(-h);
    const float ratio = t / t_prev_0;
    const float coeff = alpha_t * phi_1;
    const float half_coeff = 0.5f * coeff;
    ParallelFor(n,
                [=](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const float delta = model_prev_0[i] - model_prev_1[i];
                        const float d1 = inv_r0 * delta;
                        const float a = ratio * x[i];
                        const float b = coeff * model_prev_0[i];
                        const float c = half_coeff * d1;
                        const float ab = a - b;
                        x[i] = ab - c;
                    }
                });
}

}  // namespace din::image_gen
