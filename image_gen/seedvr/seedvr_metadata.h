// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "seedvr_pipeline.h"
#ifdef DIN_SEEDVR_NATIVE_VAE
#include "seedvr_vae.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "nvtx_helper.h"
#include "ort_session.h"
#include "rng.h"
#include "rng_cuda.h"
#include "seedvr_cuda_kernels.h"
#include "seedvr_tiling.h"
#include <onnxruntime_run_options_config_keys.h>

namespace din::seedvr::detail
{
using Fp16 = Ort::Float16_t;
namespace fs = std::filesystem;

constexpr int64_t kLatentChannels = 16;
constexpr int64_t kContextTokens = 58;
constexpr int64_t kContextWidth = 5120;
constexpr int64_t kRopePairs = 63;
constexpr int64_t kPatchSize = 2;
constexpr double kReferenceArea = 45.0 * 80.0;
constexpr int64_t kMaxTemporalWindow = 30;
constexpr int64_t kProfileMinImageSize = 32;
constexpr int64_t kProfileMaxLongSide = 3840;
constexpr int64_t kProfileMaxShortSide = 2160;

inline void CheckCuda(cudaError_t status, const char* expression)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string{"CUDA call failed: "} + expression + ": " + cudaGetErrorString(status));
    }
}

#define SEEDVR_CUDA_CHECK(call) CheckCuda((call), #call)

struct Window
{
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t h0 = 0;
    int64_t h1 = 0;
    int64_t w0 = 0;
    int64_t w1 = 0;
};

inline std::vector<Window> MakeWindows(int64_t height, int64_t width, bool shifted, int64_t temporal = 1)
{
    constexpr int64_t requested_t = 4;
    constexpr int64_t requested_h = 3;
    constexpr int64_t requested_w = 3;
    const double scale = std::sqrt(kReferenceArea / static_cast<double>(height * width));
    const int64_t resized_h = static_cast<int64_t>(std::nearbyint(height * scale));
    const int64_t resized_w = static_cast<int64_t>(std::nearbyint(width * scale));
    const int64_t wh = (resized_h + requested_h - 1) / requested_h;
    const int64_t ww = (resized_w + requested_w - 1) / requested_w;
    const int64_t wt = (std::min(temporal, kMaxTemporalWindow) + requested_t - 1) / requested_t;

    const double st = shifted && wt < temporal ? 0.5 : 0.0;
    const double sh = shifted && wh < height ? 0.5 : 0.0;
    const double sw = shifted && ww < width ? 0.5 : 0.0;
    const int64_t nt = st > 0.0 ? static_cast<int64_t>(std::ceil((temporal - st) / wt)) + 1
                                : static_cast<int64_t>(std::ceil(static_cast<double>(temporal) / wt));
    const int64_t nh = sh > 0.0 ? static_cast<int64_t>(std::ceil((height - sh) / wh)) + 1
                                : static_cast<int64_t>(std::ceil(static_cast<double>(height) / wh));
    const int64_t nw = sw > 0.0 ? static_cast<int64_t>(std::ceil((width - sw) / ww)) + 1
                                : static_cast<int64_t>(std::ceil(static_cast<double>(width) / ww));

    std::vector<Window> windows;
    for (int64_t iw = 0; iw < nw; ++iw)
    {
        for (int64_t ih = 0; ih < nh; ++ih)
        {
            for (int64_t it = 0; it < nt; ++it)
            {
                const auto begin = [](int64_t index, double shift, int64_t size)
                {
                    return std::max(static_cast<int64_t>((index - shift) * size), int64_t{0});
                };
                const auto end = [](int64_t index, double shift, int64_t size, int64_t limit)
                {
                    return std::min(static_cast<int64_t>((index - shift + 1.0) * size), limit);
                };
                Window window{begin(it, st, wt),       end(it, st, wt, temporal), begin(ih, sh, wh),
                              end(ih, sh, wh, height), begin(iw, sw, ww),         end(iw, sw, ww, width)};
                if (window.t1 > window.t0 && window.h1 > window.h0 && window.w1 > window.w0)
                {
                    windows.push_back(window);
                }
            }
        }
    }
    return windows;
}

template <typename RopeType>
inline void AppendRope(RopeType*& output, int64_t temporal_position, int64_t height_position,
                       int64_t width_position)
{
    const std::array<int64_t, 3> positions{temporal_position, height_position, width_position};
    for (const int64_t position : positions)
    {
        for (int index = 0; index < 21; ++index)
        {
            const float frequency = std::pow(10000.0f, -static_cast<float>(2 * index) / 42.0f);
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            *output++ = RopeType(cosine);
            *output++ = RopeType(-sine);
            *output++ = RopeType(sine);
            *output++ = RopeType(cosine);
        }
    }
}

template <typename RopeType>
inline void MakeMetadata(int64_t latent_height, int64_t latent_width, bool shifted, int64_t temporal,
                         bool version_7b, Ort::IoBinding& binding, const char* prefix,
                         common::TensorBuffer<int64_t>& window_tgt_idx, common::TensorBuffer<int64_t>& window_src_idx,
                         common::TensorBuffer<int64_t>& concat_tgt_idx, common::TensorBuffer<int64_t>& concat_src_idx,
                         common::TensorBuffer<int64_t>& padded_flat_idx,
                         common::TensorBuffer<int64_t>& padded_gather_idx, common::TensorBuffer<bool>& valid_keys,
                         common::TensorBuffer<RopeType>& freqs)
{
    if (latent_height < 2 || latent_width < 2 || latent_height % 2 != 0 || latent_width % 2 != 0)
    {
        throw std::invalid_argument("SeedVR latent dimensions must be positive even values");
    }
    const int64_t height = latent_height / kPatchSize;
    const int64_t width = latent_width / kPatchSize;
    const int64_t patched_tokens = temporal * height * width;
    const std::vector<Window> windows = MakeWindows(height, width, shifted, temporal);

    auto* wt = window_tgt_idx.HostData();
    auto* ws = window_src_idx.HostData();
    auto* ct = concat_tgt_idx.HostData();
    auto* cs = concat_src_idx.HostData();
    auto* pf = padded_flat_idx.HostData();
    auto* pg = padded_gather_idx.HostData();
    auto* valid = valid_keys.HostData();
    auto* rope = freqs.HostData();
    int64_t concat_index = 0;

    int64_t window_order_video_index = 0;
    for (const Window& window : windows)
    {
        const int64_t window_t = window.t1 - window.t0;
        const int64_t window_h = window.h1 - window.h0;
        const int64_t window_w = window.w1 - window.w0;
        for (int64_t t = window.t0; t < window.t1; ++t)
        {
            for (int64_t h = window.h0; h < window.h1; ++h)
            {
                for (int64_t w = window.w0; w < window.w1; ++w)
                {
                    wt[window_order_video_index] = (t * height + h) * width + w;
                    ct[concat_index++] = window_order_video_index++;
                    if (version_7b)
                    {
                        // Pixel RoPE: ten frequency pairs per axis, local positions in [-1, 1].
                        const std::array<int64_t, 3> positions{t - window.t0, h - window.h0, w - window.w0};
                        const std::array<int64_t, 3> sizes{window_t, window_h, window_w};
                        for (size_t axis = 0; axis < positions.size(); ++axis)
                        {
                            const float position = sizes[axis] == 1 ? -1.0f :
                                -1.0f + 2.0f * static_cast<float>(positions[axis]) / static_cast<float>(sizes[axis] - 1);
                            for (int index = 0; index < 10; ++index)
                            {
                                const float frequency = (1.0f + 127.0f * static_cast<float>(index) / 9.0f) *
                                                        static_cast<float>(std::acos(-1.0));
                                const float angle = position * frequency;
                                const float cosine = std::cos(angle), sine = std::sin(angle);
                                *rope++ = RopeType(cosine);
                                *rope++ = RopeType(-sine);
                                *rope++ = RopeType(sine);
                                *rope++ = RopeType(cosine);
                            }
                        }
                    }
                    else
                        AppendRope(rope, kContextTokens + (t - window.t0), h - window.h0, w - window.w0);
                }
            }
        }
        for (int64_t text = 0; text < kContextTokens; ++text)
        {
            ct[concat_index++] = patched_tokens + text;
            if (version_7b)
                for (int pair = 0; pair < 30; ++pair)
                {
                    *rope++ = RopeType(1.0f);
                    *rope++ = RopeType(0.0f);
                    *rope++ = RopeType(0.0f);
                    *rope++ = RopeType(1.0f);
                }
            else
                AppendRope(rope, text, text, text);
        }
    }
    if (window_order_video_index != patched_tokens)
    {
        throw std::runtime_error("SeedVR window metadata did not cover each patched token exactly once");
    }
    for (int64_t index = 0; index < patched_tokens; ++index)
    {
        ws[wt[index]] = index;
    }

    std::iota(cs, cs + concat_index, int64_t{0});
    std::stable_sort(cs, cs + concat_index,
                     [&](int64_t lhs, int64_t rhs)
                     {
                         return ct[lhs] < ct[rhs];
                     });

    int64_t max_length = 0;
    for (const Window& window : windows)
    {
        const int64_t length =
            (window.t1 - window.t0) * (window.h1 - window.h0) * (window.w1 - window.w0) + kContextTokens;
        max_length = std::max(max_length, length);
    }
    int64_t packed_offset = 0;
    for (size_t index = 0; index < windows.size(); ++index)
    {
        const Window& window = windows[index];
        const int64_t length =
            (window.t1 - window.t0) * (window.h1 - window.h0) * (window.w1 - window.w0) + kContextTokens;
        for (int64_t position = 0; position < max_length; ++position)
        {
            const bool is_valid = position < length;
            *pg++ = is_valid ? packed_offset + position : 0;
            *valid++ = is_valid;
            if (is_valid)
            {
                *pf++ = static_cast<int64_t>(index) * max_length + position;
            }
        }
        packed_offset += length;
    }
    if (packed_offset != concat_index)
    {
        throw std::runtime_error("SeedVR padded attention metadata has an invalid token count");
    }
    const std::string root(prefix);
    binding.BindInput((root + "_window_tgt_idx").c_str(), window_tgt_idx.BindingValue());
    window_tgt_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_window_src_idx").c_str(), window_src_idx.BindingValue());
    window_src_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_concat_tgt_idx").c_str(), concat_tgt_idx.BindingValue());
    concat_tgt_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_concat_src_idx").c_str(), concat_src_idx.BindingValue());
    concat_src_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_padded_flat_idx").c_str(), padded_flat_idx.BindingValue());
    padded_flat_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_padded_gather_idx").c_str(), padded_gather_idx.BindingValue());
    padded_gather_idx.CopyAsyncToDevice();
    binding.BindInput((root + "_valid_keys").c_str(), valid_keys.BindingValue());
    valid_keys.CopyAsyncToDevice();
    binding.BindInput((root + "_freqs").c_str(), freqs.BindingValue());
    freqs.CopyAsyncToDevice();
}

inline std::string ShapeString(std::initializer_list<int64_t> dimensions)
{
    std::string result;
    for (const int64_t dimension : dimensions)
    {
        if (!result.empty())
        {
            result += 'x';
        }
        result += std::to_string(dimension);
    }
    return result;
}

struct MetadataDimensions
{
    int64_t video_tokens;
    int64_t attention_tokens;
    int64_t padded_tokens;
    int64_t window_count;
    int64_t max_window_length;
};

inline MetadataDimensions ProfileMetadata(int64_t image_height, int64_t image_width, bool shifted, int64_t temporal = 1)
{
    const int64_t patch_height = image_height / (8 * kPatchSize);
    const int64_t patch_width = image_width / (8 * kPatchSize);
    const auto windows = MakeWindows(patch_height, patch_width, shifted, temporal);
    int64_t max_window_length = 0;
    for (const auto& window : windows)
    {
        const int64_t length =
            (window.t1 - window.t0) * (window.h1 - window.h0) * (window.w1 - window.w0) + kContextTokens;
        max_window_length = std::max(max_window_length, length);
    }
    const int64_t window_count = static_cast<int64_t>(windows.size());
    return {temporal * patch_height * patch_width,
            temporal * patch_height * patch_width + window_count * kContextTokens, window_count * max_window_length,
            window_count, max_window_length};
}

inline std::string DitProfile(int64_t image_height, int64_t image_width, const MetadataDimensions& normal,
                              const MetadataDimensions& shifted, int64_t temporal = 1, int64_t rope_pairs = kRopePairs)
{
    const auto pack = [rope_pairs](std::string_view prefix, const MetadataDimensions& metadata)
    {
        return std::string{prefix} + "_window_tgt_idx:" + ShapeString({metadata.video_tokens}) + "," +
               std::string{prefix} + "_window_src_idx:" + ShapeString({metadata.video_tokens}) + "," +
               std::string{prefix} + "_concat_tgt_idx:" + ShapeString({metadata.attention_tokens}) + "," +
               std::string{prefix} + "_concat_src_idx:" + ShapeString({metadata.attention_tokens}) + "," +
               std::string{prefix} + "_padded_flat_idx:" + ShapeString({metadata.attention_tokens}) + "," +
               std::string{prefix} + "_padded_gather_idx:" + ShapeString({metadata.padded_tokens}) + "," +
               std::string{prefix} + "_valid_keys:" + ShapeString({metadata.window_count, metadata.max_window_length}) +
               "," + std::string{prefix} + "_freqs:" + ShapeString({metadata.attention_tokens, rope_pairs, 2, 2});
    };
    const int64_t latent_height = image_height / 8;
    const int64_t latent_width = image_width / 8;
    return "latent:" + ShapeString({1, kLatentChannels, temporal, latent_height, latent_width}) +
           ",timestep:1,context:" + ShapeString({1, kContextTokens, kContextWidth}) +
           ",condition:" + ShapeString({1, kLatentChannels + 1, temporal, latent_height, latent_width}) + "," +
           pack("normal", normal) + "," + pack("shifted", shifted);
}

inline common::ModelProfile RangedDitProfile(bool video, int64_t rope_pairs = kRopePairs)
{
    struct Shape { int64_t frames, height, width; };
    const struct { Shape minimum, optimum, maximum; } range{
        {video ? static_cast<int64_t>(kVideoFrameRange.min) : 1, kProfileMinImageSize, kProfileMinImageSize},
        {video ? static_cast<int64_t>(kVideoFrameRange.min) : 1, 720, 1280},
        {video ? static_cast<int64_t>(kVideoFrameRange.max) : 1, kProfileMaxLongSide, kProfileMaxLongSide}
    };
    // A rectangular envelope covers both 4K orientations with one context.
    // Actual requests still obey the existing long/short-side limits.
    const auto bounds = [&](bool shifted)
    {
        auto low = ProfileMetadata(range.minimum.height, range.minimum.width, shifted, (range.minimum.frames - 1) / 4 + 1);
        auto high = low;
        // Windows are a Cartesian product of independent temporal/spatial axes.
        // Scan their extrema separately, retaining the existing window geometry
        // without repeating the large spatial scan for every temporal length.
        auto spatial_low = ProfileMetadata(range.minimum.height, range.minimum.width, shifted);
        auto spatial_high = spatial_low;
        for (int64_t h = range.minimum.height; h <= range.maximum.height; h += 16)
            for (int64_t w = range.minimum.width; w <= range.maximum.width; w += 16)
            {
                const auto d = ProfileMetadata(h, w, shifted);
                spatial_low.window_count = std::min(spatial_low.window_count, d.window_count);
                spatial_high.window_count = std::max(spatial_high.window_count, d.window_count);
                spatial_low.max_window_length = std::min(spatial_low.max_window_length, d.max_window_length);
                spatial_high.max_window_length = std::max(spatial_high.max_window_length, d.max_window_length);
            }
        auto temporal_low = ProfileMetadata(16, 16, shifted, (range.minimum.frames - 1) / 4 + 1);
        auto temporal_high = temporal_low;
        for (int64_t t = range.minimum.frames; t <= range.maximum.frames; t += 4)
        {
            const auto d = ProfileMetadata(16, 16, shifted, (t - 1) / 4 + 1);
            temporal_low.window_count = std::min(temporal_low.window_count, d.window_count);
            temporal_high.window_count = std::max(temporal_high.window_count, d.window_count);
            temporal_low.max_window_length = std::min(temporal_low.max_window_length, d.max_window_length);
            temporal_high.max_window_length = std::max(temporal_high.max_window_length, d.max_window_length);
        }
        low.window_count = spatial_low.window_count * temporal_low.window_count;
        high.window_count = spatial_high.window_count * temporal_high.window_count;
        low.max_window_length = (spatial_low.max_window_length - kContextTokens) *
                                (temporal_low.max_window_length - kContextTokens) + kContextTokens;
        high.max_window_length = (spatial_high.max_window_length - kContextTokens) *
                                 (temporal_high.max_window_length - kContextTokens) + kContextTokens;
        high.video_tokens = ((range.maximum.frames - 1) / 4 + 1) * (range.maximum.height / 16) * (range.maximum.width / 16);
        for (auto* d : {&low, &high})
        {
            d->attention_tokens = d->video_tokens + d->window_count * kContextTokens;
            d->padded_tokens = d->window_count * d->max_window_length;
        }
        return std::pair{low, high};
    };
    const auto calculate_bounds = [&]
    {
        const auto [normal_min, normal_max] = bounds(false);
        const auto [shifted_min, shifted_max] = bounds(true);
        return std::array{normal_min, normal_max, shifted_min, shifted_max};
    };
    // Geometry bounds depend only on image/video mode, not the current shape,
    // model precision or RoPE width. Scan once per mode across pipeline instances.
    const auto& [normal_min, normal_max, shifted_min, shifted_max] =
        [&]() -> const std::array<MetadataDimensions, 4>&
        {
            if (video)
            {
                static const auto value = calculate_bounds();
                return value;
            }
            static const auto value = calculate_bounds();
            return value;
        }();
    common::ModelProfile profile;
    profile.min_shapes = DitProfile(range.minimum.height, range.minimum.width, normal_min, shifted_min, (range.minimum.frames - 1) / 4 + 1, rope_pairs);
    profile.opt_shapes = DitProfile(range.optimum.height, range.optimum.width,
        ProfileMetadata(range.optimum.height, range.optimum.width, false, (range.optimum.frames - 1) / 4 + 1),
        ProfileMetadata(range.optimum.height, range.optimum.width, true, (range.optimum.frames - 1) / 4 + 1), (range.optimum.frames - 1) / 4 + 1, rope_pairs);
    profile.max_shapes = DitProfile(range.maximum.height, range.maximum.width, normal_max, shifted_max, (range.maximum.frames - 1) / 4 + 1, rope_pairs);
    profile.embed_ep_context = false;
    profile.cache_subpath = std::string(video ? "video_" : "image_") + "dit_range_v1";
    return profile;
}

inline common::ModelProfile RangedVaeProfile(bool video, bool encode)
{
    const auto shape = [&](int64_t frames, int64_t height, int64_t width)
    {
        if (!encode)
        {
            frames = (frames - 1) / 4 + 1;
            height /= 8;
            width /= 8;
        }
        const int64_t channels = encode ? 3 : kLatentChannels;
        return std::string(encode ? "image:" : "latent:") +
        (video
             ? ShapeString({1, channels, frames, height, width})
             : ShapeString({1, channels, height, width}));
    };
    common::ModelProfile profile;
    const int64_t min_frames = video ? kVideoFrameRange.min : 1;
    const int64_t max_frames = video ? kVideoFrameRange.max : 1;
    profile.min_shapes = shape(min_frames, kProfileMinImageSize, kProfileMinImageSize);
    profile.opt_shapes = shape(min_frames, 720, 1280);
    // The square envelope includes landscape and portrait 4K. Pipeline requests
    // still enforce the long/short-side limits and valid padded/temporal steps.
    profile.max_shapes = shape(max_frames, kProfileMaxLongSide, kProfileMaxLongSide);
    profile.cache_subpath = std::string(video ? "video_" : "image_") +
        (encode ? "vae_encoder_range_v1" : "vae_decoder_range_v1");
    profile.embed_ep_context = false;
    return profile;
}

// Exported DiT/video VAE symbols use the first input sharing a dimension:
// <input>_dim<axis>. Adding every input alias also covers shared metadata axes.
inline common::ModelProfile ExactProfile(std::string shapes)
{
    common::ModelProfile profile;
    profile.min_shapes = profile.opt_shapes = profile.max_shapes = shapes;
    profile.embed_ep_context = false;
    profile.force_recompile = true;
    std::istringstream inputs(shapes);
    for (std::string input; std::getline(inputs, input, ',');)
    {
        const auto colon = input.find(':');
        const auto name = input.substr(0, colon);
        std::istringstream dimensions(input.substr(colon + 1));
        size_t axis = 0;
        for (std::string dimension; std::getline(dimensions, dimension, 'x'); ++axis)
            profile.free_dimension_overrides.emplace_back(name + "_dim" + std::to_string(axis),
                                                          std::stoll(dimension));
    }
    return profile;
}

inline common::ModelProfile ExactVaeProfile(bool video, bool encode, int64_t frames, int64_t height, int64_t width)
{
    // Decoder arguments are already latent dimensions (including tile geometry).
    const auto name = encode ? "image" : "latent";
    const int64_t channels = encode ? 3 : kLatentChannels;
    auto profile = ExactProfile(std::string(name) + ":" +
        (video ? ShapeString({1, channels, frames, height, width}) : ShapeString({1, channels, height, width})));
    if (!video)
    {
        // Image VAEs retain torch.export's derived symbolic names.
        const std::string multiple = encode ? "16*" : "2*";
        profile.free_dimension_overrides.emplace_back(multiple + name + "_height_units", height);
        profile.free_dimension_overrides.emplace_back(multiple + name + "_width_units", width);
    }
    return profile;
}

inline common::ModelProfile ExactDitProfile(int64_t height, int64_t width, int64_t latent_frames,
                                            int64_t rope_pairs = kRopePairs)
{
    return ExactProfile(DitProfile(height, width, ProfileMetadata(height, width, false, latent_frames),
                                   ProfileMetadata(height, width, true, latent_frames), latent_frames, rope_pairs));
}

inline void ConfigureCache(common::ModelProfile& profile, const fs::path& model)
{
    // Precision and image/video variants are part of the model filename. Neither
    // execution shape nor ranged/static policy belongs in this namespace.
    profile.stable_cache_path = (fs::absolute(model).lexically_normal().parent_path().filename() / model.stem()).
        generic_string();
}

inline void RequireFile(const fs::path& path)
{
    if (!fs::is_regular_file(path))
    {
        throw std::runtime_error("Missing SeedVR runtime file: " + path.string());
    }
}

inline std::vector<Fp16> LoadContext(const fs::path& path)
{
    RequireFile(path);
    constexpr size_t count = static_cast<size_t>(kContextTokens * kContextWidth);
    if (fs::file_size(path) != count * sizeof(Fp16))
    {
        throw std::runtime_error("SeedVR positive context has unexpected size: " + path.string());
    }
    std::vector<Fp16> result(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size() * sizeof(Fp16)));
    if (!stream)
    {
        throw std::runtime_error("Failed to read SeedVR positive context: " + path.string());
    }
    return result;
}

}  // namespace din::seedvr::detail
