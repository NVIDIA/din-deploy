// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "seedvr_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "io/image.h"
#include "ort_session.h"
#include "seedvr_cuda_kernels.h"
#include <lodepng.h>
#include <onnxruntime_run_options_config_keys.h>

namespace din::seedvr
{
    namespace
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

        void CheckCuda(cudaError_t status, const char* expression)
        {
            if (status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string{"CUDA call failed: "} + expression + ": " + cudaGetErrorString(status));
            }
        }

#define SEEDVR_CUDA_CHECK(call) CheckCuda((call), #call)

        struct RawImage
        {
            std::vector<float> hwc;
            size_t height = 0;
            size_t width = 0;
        };

        struct Window
        {
            int64_t t0 = 0;
            int64_t t1 = 0;
            int64_t h0 = 0;
            int64_t h1 = 0;
            int64_t w0 = 0;
            int64_t w1 = 0;
        };

        struct MetadataPack
        {
            std::vector<int64_t> window_tgt_idx;
            std::vector<int64_t> window_shape;
            std::vector<int64_t> concat_tgt_idx;
            std::vector<int64_t> concat_src_idx;
            std::vector<Fp16> freqs;

            size_t WindowCount() const
            {
                return window_shape.size() / 3;
            }
        };

        RawImage LoadPng(const fs::path& path)
        {
            std::vector<unsigned char> rgba;
            unsigned int width = 0;
            unsigned int height = 0;
            const unsigned int error = lodepng::decode(rgba, width, height, path.string());
            if (error != 0)
            {
                throw std::runtime_error("Failed to decode " + path.string() + ": " + lodepng_error_text(error));
            }

            RawImage image;
            image.height = height;
            image.width = width;
            image.hwc.resize(static_cast<size_t>(height) * width * 3);
            for (size_t pixel = 0; pixel < static_cast<size_t>(height) * width; ++pixel)
            {
                image.hwc[pixel * 3 + 0] = static_cast<float>(rgba[pixel * 4 + 0]) / 255.0f;
                image.hwc[pixel * 3 + 1] = static_cast<float>(rgba[pixel * 4 + 1]) / 255.0f;
                image.hwc[pixel * 3 + 2] = static_cast<float>(rgba[pixel * 4 + 2]) / 255.0f;
            }
            return image;
        }

        float CubicWeight(float x)
        {
            constexpr float a = -0.75f;
            x = std::abs(x);
            if (x <= 1.0f)
            {
                return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
            }
            if (x < 2.0f)
            {
                return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
            }
            return 0.0f;
        }

        RawImage ResizeBicubic(const RawImage& source, size_t output_height, size_t output_width)
        {
            RawImage output;
            output.height = output_height;
            output.width = output_width;
            output.hwc.resize(output_height * output_width * 3);
            const float scale_y = static_cast<float>(source.height) / static_cast<float>(output_height);
            const float scale_x = static_cast<float>(source.width) / static_cast<float>(output_width);

            for (size_t y = 0; y < output_height; ++y)
            {
                const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                const int y_base = static_cast<int>(std::floor(source_y));
                for (size_t x = 0; x < output_width; ++x)
                {
                    const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    const int x_base = static_cast<int>(std::floor(source_x));
                    for (size_t channel = 0; channel < 3; ++channel)
                    {
                        float value = 0.0f;
                        float weight_sum = 0.0f;
                        for (int ky = -1; ky <= 2; ++ky)
                        {
                            const int sy = std::clamp(y_base + ky, 0, static_cast<int>(source.height) - 1);
                            const float wy = CubicWeight(source_y - static_cast<float>(y_base + ky));
                            for (int kx = -1; kx <= 2; ++kx)
                            {
                                const int sx = std::clamp(x_base + kx, 0, static_cast<int>(source.width) - 1);
                                const float weight = wy * CubicWeight(source_x - static_cast<float>(x_base + kx));
                                value += source.hwc[(static_cast<size_t>(sy) * source.width + sx) * 3 + channel] *
                                    weight;
                                weight_sum += weight;
                            }
                        }
                        output.hwc[(y * output_width + x) * 3 + channel] = std::clamp(value / weight_sum, 0.0f, 1.0f);
                    }
                }
            }
            return output;
        }

        std::vector<Fp16> PadToNchwFp16(const RawImage& image, size_t padded_height, size_t padded_width)
        {
            std::vector<Fp16> result(3 * padded_height * padded_width, Fp16(0.0f));
            const size_t plane = padded_height * padded_width;
            for (size_t y = 0; y < image.height; ++y)
            {
                for (size_t x = 0; x < image.width; ++x)
                {
                    for (size_t channel = 0; channel < 3; ++channel)
                    {
                        result[channel * plane + y * padded_width + x] = Fp16(
                            image.hwc[(y * image.width + x) * 3 + channel]);
                    }
                }
            }
            return result;
        }

        std::vector<Window> MakeWindows(int64_t height, int64_t width, bool shifted)
        {
            constexpr int64_t temporal = 1;
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
            const int64_t nt = st > 0.0 ? static_cast<int64_t>(std::ceil((temporal - st) / wt)) + 1 : 1;
            const int64_t nh = sh > 0.0
                                   ? static_cast<int64_t>(std::ceil((height - sh) / wh)) + 1
                                   : static_cast<int64_t>(std::ceil(static_cast<double>(height) / wh));
            const int64_t nw = sw > 0.0
                                   ? static_cast<int64_t>(std::ceil((width - sw) / ww)) + 1
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
                        Window window{
                            begin(it, st, wt), end(it, st, wt, temporal), begin(ih, sh, wh),
                            end(ih, sh, wh, height), begin(iw, sw, ww), end(iw, sw, ww, width)
                        };
                        if (window.t1 > window.t0 && window.h1 > window.h0 && window.w1 > window.w0)
                        {
                            windows.push_back(window);
                        }
                    }
                }
            }
            return windows;
        }

        void AppendRope(std::vector<Fp16>& output, int64_t temporal_position, int64_t height_position,
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
                    output.emplace_back(cosine);
                    output.emplace_back(-sine);
                    output.emplace_back(sine);
                    output.emplace_back(cosine);
                }
            }
        }

        MetadataPack MakeMetadata(int64_t latent_height, int64_t latent_width, bool shifted)
        {
            if (latent_height < 2 || latent_width < 2 || latent_height % 2 != 0 || latent_width % 2 != 0)
            {
                throw std::invalid_argument("SeedVR latent dimensions must be positive even values");
            }
            const int64_t height = latent_height / kPatchSize;
            const int64_t width = latent_width / kPatchSize;
            const int64_t patched_tokens = height * width;
            const std::vector<Window> windows = MakeWindows(height, width, shifted);

            MetadataPack pack;
            pack.window_tgt_idx.reserve(static_cast<size_t>(patched_tokens));
            pack.window_shape.reserve(windows.size() * 3);
            pack.concat_tgt_idx.reserve(static_cast<size_t>(patched_tokens) + windows.size() * kContextTokens);
            pack.freqs.reserve(
                (static_cast<size_t>(patched_tokens) + windows.size() * kContextTokens) * kRopePairs * 4);

            int64_t window_order_video_index = 0;
            for (const Window& window : windows)
            {
                const int64_t window_t = window.t1 - window.t0;
                const int64_t window_h = window.h1 - window.h0;
                const int64_t window_w = window.w1 - window.w0;
                pack.window_shape.insert(pack.window_shape.end(), {window_t, window_h, window_w});
                for (int64_t t = window.t0; t < window.t1; ++t)
                {
                    for (int64_t h = window.h0; h < window.h1; ++h)
                    {
                        for (int64_t w = window.w0; w < window.w1; ++w)
                        {
                            pack.window_tgt_idx.push_back((t * height + h) * width + w);
                            pack.concat_tgt_idx.push_back(window_order_video_index++);
                            AppendRope(pack.freqs, kContextTokens + (t - window.t0), h - window.h0, w - window.w0);
                        }
                    }
                }
                for (int64_t text = 0; text < kContextTokens; ++text)
                {
                    pack.concat_tgt_idx.push_back(patched_tokens + text);
                    AppendRope(pack.freqs, text, text, text);
                }
            }
            if (window_order_video_index != patched_tokens)
            {
                throw std::runtime_error("SeedVR window metadata did not cover each patched token exactly once");
            }

            pack.concat_src_idx.resize(pack.concat_tgt_idx.size());
            std::iota(pack.concat_src_idx.begin(), pack.concat_src_idx.end(), int64_t{0});
            std::stable_sort(pack.concat_src_idx.begin(), pack.concat_src_idx.end(),
                             [&](int64_t lhs, int64_t rhs)
                             {
                                 return pack.concat_tgt_idx[static_cast<size_t>(lhs)] <
                                     pack.concat_tgt_idx[static_cast<size_t>(rhs)];
                             });
            return pack;
        }

        std::string ShapeString(std::initializer_list<int64_t> dimensions)
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

        std::string DitProfile(int64_t latent_height, int64_t latent_width, const MetadataPack& normal,
                               const MetadataPack& shifted)
        {
            const auto pack = [](std::string_view prefix, const MetadataPack& metadata)
            {
                const int64_t attention_tokens = static_cast<int64_t>(metadata.concat_tgt_idx.size());
                return std::string{prefix} +
                    "_window_tgt_idx:" + ShapeString({static_cast<int64_t>(metadata.window_tgt_idx.size())}) + "," +
                    std::string{prefix} + "_window_shape:" + ShapeString({
                        static_cast<int64_t>(metadata.WindowCount()), 3
                    }) +
                    "," + std::string{prefix} + "_concat_tgt_idx:" + ShapeString({attention_tokens}) + "," +
                    std::string{prefix} + "_concat_src_idx:" + ShapeString({attention_tokens}) + "," + std::string{
                        prefix
                    } +
                    "_freqs:" + ShapeString({attention_tokens, kRopePairs, 2, 2});
            };
            return "latent:" + ShapeString({1, kLatentChannels, 1, latent_height, latent_width}) +
                ",timestep:1,context:" + ShapeString({1, kContextTokens, kContextWidth}) +
                ",condition:" + ShapeString({1, kLatentChannels + 1, 1, latent_height, latent_width}) + "," +
                pack("normal", normal) + "," + pack("shifted", shifted);
        }

        common::ModelProfile ExactProfile(std::string shapes, std::string cache_subpath)
        {
            common::ModelProfile profile;
            profile.min_shapes = shapes;
            profile.opt_shapes = shapes;
            profile.max_shapes = std::move(shapes);
            profile.cache_subpath = std::move(cache_subpath);
            profile.enable_cuda_graph = false;
            profile.embed_ep_context = false;
            profile.skip_compile = true;
            profile.extra_ep_options.emplace_back("nv_length_aux_stream_array", "0");
            return profile;
        }

        void RequireFile(const fs::path& path)
        {
            if (!fs::is_regular_file(path))
            {
                throw std::runtime_error("Missing SeedVR runtime file: " + path.string());
            }
        }

        std::vector<Fp16> LoadContext(const fs::path& path)
        {
            RequireFile(path);
            constexpr size_t count = static_cast<size_t>(kContextTokens * kContextWidth);
            if (fs::file_size(path) != count * sizeof(Fp16))
            {
                throw std::runtime_error("SeedVR positive context has unexpected size: " + path.string());
            }
            std::vector<Fp16> result(count);
            std::ifstream stream(path, std::ios::binary);
            stream.read(reinterpret_cast<char*>(result.data()),
                        static_cast<std::streamsize>(result.size() * sizeof(Fp16)));
            if (!stream)
            {
                throw std::runtime_error("Failed to read SeedVR positive context: " + path.string());
            }
            return result;
        }

        template <typename T>
        void FillBuffer(common::TensorBuffer<T>& buffer, const std::vector<T>& values)
        {
            std::copy(values.begin(), values.end(), buffer.HostData());
            buffer.CopyAsyncToDevice();
        }

        void BindMetadata(Ort::IoBinding& binding, const char* prefix, MetadataPack& pack,
                          common::TensorBuffer<int64_t>& window_tgt_idx, common::TensorBuffer<int64_t>& window_shape,
                          common::TensorBuffer<int64_t>& concat_tgt_idx, common::TensorBuffer<int64_t>& concat_src_idx,
                          common::TensorBuffer<Fp16>& freqs)
        {
            const std::string root(prefix);
            binding.BindInput((root + "_window_tgt_idx").c_str(), window_tgt_idx.BindingValue());
            binding.BindInput((root + "_window_shape").c_str(), window_shape.BindingValue());
            binding.BindInput((root + "_concat_tgt_idx").c_str(), concat_tgt_idx.BindingValue());
            binding.BindInput((root + "_concat_src_idx").c_str(), concat_src_idx.BindingValue());
            binding.BindInput((root + "_freqs").c_str(), freqs.BindingValue());
            FillBuffer(window_tgt_idx, pack.window_tgt_idx);
            FillBuffer(window_shape, pack.window_shape);
            FillBuffer(concat_tgt_idx, pack.concat_tgt_idx);
            FillBuffer(concat_src_idx, pack.concat_src_idx);
            FillBuffer(freqs, pack.freqs);
        }
    } // namespace

    class Pipeline::Impl
    {
    public:
        explicit Impl(Config config)
            : config_(std::move(config))
              , env_(ORT_LOGGING_LEVEL_WARNING, "SeedVR2")
        {
        }

        Image Run()
        {
            if (config_.scale <= 0.0f)
            {
                throw std::invalid_argument("SeedVR scale must be positive");
            }
            RequireFile(config_.input_path);
            const RawImage source = LoadPng(config_.input_path);
            const size_t resized_height =
                std::max<size_t>(2, static_cast<size_t>(std::nearbyint(source.height * config_.scale)));
            const size_t resized_width =
                std::max<size_t>(2, static_cast<size_t>(std::nearbyint(source.width * config_.scale)));
            const RawImage resized = ResizeBicubic(source, resized_height, resized_width);
            const size_t padded_height = (resized_height + 15) / 16 * 16;
            const size_t padded_width = (resized_width + 15) / 16 * 16;
            const int64_t latent_height = static_cast<int64_t>(padded_height / 8);
            const int64_t latent_width = static_cast<int64_t>(padded_width / 8);
            MetadataPack normal = MakeMetadata(latent_height, latent_width, false);
            MetadataPack shifted = MakeMetadata(latent_height, latent_width, true);
            std::cout << "Input " << source.width << 'x' << source.height << " -> resized " << resized_width << 'x'
                << resized_height << " -> padded " << padded_width << 'x' << padded_height << "\n"
                << "Latent " << latent_width << 'x' << latent_height << ", normal windows=" << normal.WindowCount()
                << ", shifted windows=" << shifted.WindowCount() << std::endl;

            const fs::path encoder_path = config_.model_dir / "seedvr2_ema_vae_encoder.onnx";
            const fs::path dit_path = config_.model_dir / "seedvr2_3b_fp16_dit.onnx";
            const fs::path decoder_path = config_.model_dir / "seedvr2_ema_vae_decoder.onnx";
            const fs::path context_path = config_.model_dir / "seedvr2_3b_fp16_positive_context.bin";
            RequireFile(encoder_path);
            RequireFile(dit_path);
            RequireFile(decoder_path);
            const std::vector<Fp16> context_values = LoadContext(context_path);

            const auto trt_device = common::RegisterTensorRTRTXProvider(env_);
            const int cuda_device = common::ChooseCudaDeviceOrdinal(trt_device, "SEEDVR_CUDA_DEVICE_ID");
            SEEDVR_CUDA_CHECK(cudaSetDevice(cuda_device));
            compute_stream_ = common::CreateTensorRTRTXComputeStream(env_);
            auto* stream = &compute_stream_;
            const common::EpContextOptions ep_context{.output_dir = config_.ep_context_dir.string()};

            common::OrtRunner encoder(env_, encoder_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                                      ExactProfile("image:" + ShapeString({
                                                       1, 3, static_cast<int64_t>(padded_height),
                                                       static_cast<int64_t>(padded_width)
                                                   }),
                                                   "vae_encoder"),
                                      stream);
            common::OrtRunner dit(env_, dit_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                                  ExactProfile(DitProfile(latent_height, latent_width, normal, shifted), "dit"),
                                  stream);
            common::OrtRunner decoder(
                env_, decoder_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                ExactProfile("latent:" + ShapeString({1, kLatentChannels, latent_height, latent_width}), "vae_decoder"),
                stream);
            if (!encoder.HasDeviceIo() || !dit.HasDeviceIo() || !decoder.HasDeviceIo())
            {
                throw std::runtime_error("SeedVR requires CUDA-backed TensorRT RTX I/O");
            }

            const size_t spatial_size = static_cast<size_t>(latent_height * latent_width);
            const std::vector<int64_t> image_shape{
                1, 3, static_cast<int64_t>(padded_height),
                static_cast<int64_t>(padded_width)
            };
            const std::vector<int64_t> encoded_shape{1, kLatentChannels, latent_height, latent_width};
            const std::vector<int64_t> latent_shape{1, kLatentChannels, 1, latent_height, latent_width};
            const std::vector<int64_t> condition_shape{1, kLatentChannels + 1, 1, latent_height, latent_width};
            const std::vector<int64_t> output_shape{
                1, 3, static_cast<int64_t>(padded_height),
                static_cast<int64_t>(padded_width)
            };

            common::TensorBuffer<Fp16> image(encoder, image_shape, true);
            common::TensorBuffer<Fp16> encoded(encoder, encoded_shape, true);
            common::TensorBuffer<Fp16> latent(dit, latent_shape, true);
            common::TensorBuffer<Fp16> timestep(dit, {1}, true);
            common::TensorBuffer<Fp16> context(dit, {1, kContextTokens, kContextWidth}, true);
            common::TensorBuffer<Fp16> condition(dit, condition_shape, true);
            common::TensorBuffer<Fp16> velocity(dit, latent_shape, true);
            common::TensorBuffer<Fp16> sampled(decoder, encoded_shape, true);
            common::TensorBuffer<Fp16> decoded(decoder, output_shape, true);

            auto make_pack_buffers = [&](common::OrtRunner& runner, const MetadataPack& pack)
            {
                struct Buffers
                {
                    common::TensorBuffer<int64_t> window_tgt_idx;
                    common::TensorBuffer<int64_t> window_shape;
                    common::TensorBuffer<int64_t> concat_tgt_idx;
                    common::TensorBuffer<int64_t> concat_src_idx;
                    common::TensorBuffer<Fp16> freqs;
                };
                const int64_t attention = static_cast<int64_t>(pack.concat_tgt_idx.size());
                return Buffers{
                    common::TensorBuffer<int64_t>(runner, {static_cast<int64_t>(pack.window_tgt_idx.size())}, true),
                    common::TensorBuffer<int64_t>(runner, {static_cast<int64_t>(pack.WindowCount()), 3}, true),
                    common::TensorBuffer<int64_t>(runner, {attention}, true),
                    common::TensorBuffer<int64_t>(runner, {attention}, true),
                    common::TensorBuffer<Fp16>(runner, {attention, kRopePairs, 2, 2}, true)
                };
            };
            auto normal_buffers = make_pack_buffers(dit, normal);
            auto shifted_buffers = make_pack_buffers(dit, shifted);

            Ort::IoBinding encoder_io(encoder.session);
            encoder_io.BindInput("image", image.BindingValue());
            encoder_io.BindOutput("latent", encoded.BindingValue());
            Ort::IoBinding dit_io(dit.session);
            dit_io.BindInput("latent", latent.BindingValue());
            dit_io.BindInput("timestep", timestep.BindingValue());
            dit_io.BindInput("context", context.BindingValue());
            dit_io.BindInput("condition", condition.BindingValue());
            dit_io.BindOutput("denoised_latent", velocity.BindingValue());
            BindMetadata(dit_io, "normal", normal, normal_buffers.window_tgt_idx, normal_buffers.window_shape,
                         normal_buffers.concat_tgt_idx, normal_buffers.concat_src_idx, normal_buffers.freqs);
            BindMetadata(dit_io, "shifted", shifted, shifted_buffers.window_tgt_idx, shifted_buffers.window_shape,
                         shifted_buffers.concat_tgt_idx, shifted_buffers.concat_src_idx, shifted_buffers.freqs);
            Ort::IoBinding decoder_io(decoder.session);
            decoder_io.BindInput("latent", sampled.BindingValue());
            decoder_io.BindOutput("image", decoded.BindingValue());

            FillBuffer(image, PadToNchwFp16(resized, padded_height, padded_width));
            FillBuffer(context, context_values);
            timestep.HostData()[0] = Fp16(1000.0f);
            timestep.CopyAsyncToDevice();
            std::mt19937 generator(config_.seed);
            std::normal_distribution<float> normal_distribution;
            std::vector<Fp16> noise(static_cast<size_t>(kLatentChannels) * spatial_size);
            std::generate(noise.begin(), noise.end(),
                          [&]
                          {
                              return Fp16(normal_distribution(generator));
                          });
            FillBuffer(latent, noise);

            Ort::RunOptions run_options;
            run_options.SetSyncStream(compute_stream_);
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
            std::cout << "Running VAE encoder" << std::endl;
            encoder.session.Run(run_options, encoder_io);
            const auto cuda_stream = reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
            LaunchSeedVrConditionKernel(cuda_stream, encoded.BindingValue().GetTensorData<Fp16>(),
                                        condition.BindingValue().GetTensorMutableData<Fp16>(), spatial_size);
            SEEDVR_CUDA_CHECK(cudaGetLastError());
            std::cout << "Running SeedVR2 DiT" << std::endl;
            dit.session.Run(run_options, dit_io);
            LaunchSeedVrEulerKernel(cuda_stream, latent.BindingValue().GetTensorData<Fp16>(),
                                    velocity.BindingValue().GetTensorData<Fp16>(),
                                    sampled.BindingValue().GetTensorMutableData<Fp16>(), noise.size());
            SEEDVR_CUDA_CHECK(cudaGetLastError());
            std::cout << "Running VAE decoder" << std::endl;
            decoder.session.Run(run_options, decoder_io);
            auto output_ready = decoded.CopyAsyncToHostWithNotification();
            output_ready.Sync();

            Image result;
            result.height = resized_height - resized_height % 2;
            result.width = resized_width - resized_width % 2;
            result.chw.resize(3 * result.height * result.width);
            const Fp16* decoded_data = decoded.HostData();
            const size_t padded_plane = padded_height * padded_width;
            const size_t result_plane = result.height * result.width;
            for (size_t channel = 0; channel < 3; ++channel)
            {
                for (size_t y = 0; y < result.height; ++y)
                {
                    for (size_t x = 0; x < result.width; ++x)
                    {
                        result.chw[channel * result_plane + y * result.width + x] =
                            std::clamp(decoded_data[channel * padded_plane + y * padded_width + x].ToFloat(), 0.0f,
                                       1.0f);
                    }
                }
            }
            return result;
        }

    private:
        Config config_;
        Ort::Env env_;
        Ort::SyncStream compute_stream_{nullptr};
    };

    Pipeline::Pipeline(Config config)
        : impl_(std::make_unique<Impl>(std::move(config)))
    {
    }

    Pipeline::~Pipeline() = default;

    Image Pipeline::Run()
    {
        return impl_->Run();
    }
} // namespace din::seedvr
