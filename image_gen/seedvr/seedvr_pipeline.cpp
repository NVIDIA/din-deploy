// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "seedvr_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "io/image.h"
#include "ort_session.h"
#include "rng.h"
#include "rng_cuda.h"
#include "seedvr_cuda_kernels.h"
#include "seedvr_tiling.h"
#include <lodepng.h>
#include <onnxruntime_run_options_config_keys.h>

namespace din::seedvr {
    namespace {
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

        void CheckCuda(cudaError_t status, const char *expression) {
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string{"CUDA call failed: "} + expression + ": " + cudaGetErrorString(status));
            }
        }

#define SEEDVR_CUDA_CHECK(call) CheckCuda((call), #call)

        struct RawImage {
            std::vector<float> hwc;
            size_t height = 0;
            size_t width = 0;
        };

        struct Window {
            int64_t t0 = 0;
            int64_t t1 = 0;
            int64_t h0 = 0;
            int64_t h1 = 0;
            int64_t w0 = 0;
            int64_t w1 = 0;
        };

        struct MetadataPack {
            std::vector<int64_t> window_tgt_idx;
            std::vector<int64_t> window_src_idx;
            std::vector<int64_t> window_shape;
            std::vector<int64_t> concat_tgt_idx;
            std::vector<int64_t> concat_src_idx;
            std::vector<int64_t> padded_flat_idx;
            std::vector<int64_t> padded_gather_idx;
            std::vector<uint8_t> valid_keys;
            std::vector<Fp16> freqs;

            size_t WindowCount() const {
                return window_shape.size() / 3;
            }

            size_t MaxWindowLength() const {
                return valid_keys.size() / WindowCount();
            }
        };

        RawImage LoadPng(const fs::path &path) {
            std::vector<unsigned char> rgba;
            unsigned int width = 0;
            unsigned int height = 0;
            const unsigned int error = lodepng::decode(rgba, width, height, path.string());
            if (error != 0) {
                throw std::runtime_error("Failed to decode " + path.string() + ": " + lodepng_error_text(error));
            }

            RawImage image;
            image.height = height;
            image.width = width;
            image.hwc.resize(static_cast<size_t>(height) * width * 3);
            for (size_t pixel = 0; pixel < static_cast<size_t>(height) * width; ++pixel) {
                image.hwc[pixel * 3 + 0] = static_cast<float>(rgba[pixel * 4 + 0]) / 255.0f;
                image.hwc[pixel * 3 + 1] = static_cast<float>(rgba[pixel * 4 + 1]) / 255.0f;
                image.hwc[pixel * 3 + 2] = static_cast<float>(rgba[pixel * 4 + 2]) / 255.0f;
            }
            return image;
        }

        float CubicWeight(float x) {
            constexpr float a = -0.75f;
            x = std::abs(x);
            if (x <= 1.0f) {
                return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
            }
            if (x < 2.0f) {
                return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
            }
            return 0.0f;
        }

        RawImage ResizeBicubic(const RawImage &source, size_t output_height, size_t output_width) {
            RawImage output;
            output.height = output_height;
            output.width = output_width;
            output.hwc.resize(output_height * output_width * 3);
            const float scale_y = static_cast<float>(source.height) / static_cast<float>(output_height);
            const float scale_x = static_cast<float>(source.width) / static_cast<float>(output_width);

            for (size_t y = 0; y < output_height; ++y) {
                const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                const int y_base = static_cast<int>(std::floor(source_y));
                for (size_t x = 0; x < output_width; ++x) {
                    const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    const int x_base = static_cast<int>(std::floor(source_x));
                    for (size_t channel = 0; channel < 3; ++channel) {
                        float value = 0.0f;
                        float weight_sum = 0.0f;
                        for (int ky = -1; ky <= 2; ++ky) {
                            const int sy = std::clamp(y_base + ky, 0, static_cast<int>(source.height) - 1);
                            const float wy = CubicWeight(source_y - static_cast<float>(y_base + ky));
                            for (int kx = -1; kx <= 2; ++kx) {
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

        std::vector<Fp16> PadToNchwFp16(const RawImage &image, size_t padded_height, size_t padded_width) {
            std::vector<Fp16> result(3 * padded_height * padded_width, Fp16(0.0f));
            const size_t plane = padded_height * padded_width;
            for (size_t y = 0; y < image.height; ++y) {
                for (size_t x = 0; x < image.width; ++x) {
                    for (size_t channel = 0; channel < 3; ++channel) {
                        result[channel * plane + y * padded_width + x] = Fp16(
                            image.hwc[(y * image.width + x) * 3 + channel]);
                    }
                }
            }
            return result;
        }

        std::vector<float> PadToNchwFloat(const RawImage &image, size_t padded_height, size_t padded_width) {
            std::vector<float> result(3 * padded_height * padded_width, 0.0f);
            const size_t plane = padded_height * padded_width;
            for (size_t y = 0; y < image.height; ++y) {
                for (size_t x = 0; x < image.width; ++x) {
                    for (size_t channel = 0; channel < 3; ++channel) {
                        result[channel * plane + y * padded_width + x] =
                            image.hwc[(y * image.width + x) * 3 + channel];
                    }
                }
            }
            return result;
        }

        std::vector<Window> MakeWindows(int64_t height, int64_t width, bool shifted, int64_t temporal = 1) {
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
            const int64_t nh = sh > 0.0
                                   ? static_cast<int64_t>(std::ceil((height - sh) / wh)) + 1
                                   : static_cast<int64_t>(std::ceil(static_cast<double>(height) / wh));
            const int64_t nw = sw > 0.0
                                   ? static_cast<int64_t>(std::ceil((width - sw) / ww)) + 1
                                   : static_cast<int64_t>(std::ceil(static_cast<double>(width) / ww));

            std::vector<Window> windows;
            for (int64_t iw = 0; iw < nw; ++iw) {
                for (int64_t ih = 0; ih < nh; ++ih) {
                    for (int64_t it = 0; it < nt; ++it) {
                        const auto begin = [](int64_t index, double shift, int64_t size) {
                            return std::max(static_cast<int64_t>((index - shift) * size), int64_t{0});
                        };
                        const auto end = [](int64_t index, double shift, int64_t size, int64_t limit) {
                            return std::min(static_cast<int64_t>((index - shift + 1.0) * size), limit);
                        };
                        Window window{
                            begin(it, st, wt), end(it, st, wt, temporal), begin(ih, sh, wh),
                            end(ih, sh, wh, height), begin(iw, sw, ww), end(iw, sw, ww, width)
                        };
                        if (window.t1 > window.t0 && window.h1 > window.h0 && window.w1 > window.w0) {
                            windows.push_back(window);
                        }
                    }
                }
            }
            return windows;
        }

        void AppendRope(std::vector<Fp16> &output, int64_t temporal_position, int64_t height_position,
                        int64_t width_position) {
            const std::array<int64_t, 3> positions{temporal_position, height_position, width_position};
            for (const int64_t position: positions) {
                for (int index = 0; index < 21; ++index) {
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

        MetadataPack MakeMetadata(int64_t latent_height, int64_t latent_width, bool shifted, int64_t temporal = 1) {
            if (latent_height < 2 || latent_width < 2 || latent_height % 2 != 0 || latent_width % 2 != 0) {
                throw std::invalid_argument("SeedVR latent dimensions must be positive even values");
            }
            const int64_t height = latent_height / kPatchSize;
            const int64_t width = latent_width / kPatchSize;
            const int64_t patched_tokens = temporal * height * width;
            const std::vector<Window> windows = MakeWindows(height, width, shifted, temporal);

            MetadataPack pack;
            pack.window_tgt_idx.reserve(static_cast<size_t>(patched_tokens));
            pack.window_shape.reserve(windows.size() * 3);
            pack.concat_tgt_idx.reserve(static_cast<size_t>(patched_tokens) + windows.size() * kContextTokens);
            pack.freqs.reserve(
                (static_cast<size_t>(patched_tokens) + windows.size() * kContextTokens) * kRopePairs * 4);

            int64_t window_order_video_index = 0;
            for (const Window &window: windows) {
                const int64_t window_t = window.t1 - window.t0;
                const int64_t window_h = window.h1 - window.h0;
                const int64_t window_w = window.w1 - window.w0;
                pack.window_shape.insert(pack.window_shape.end(), {window_t, window_h, window_w});
                for (int64_t t = window.t0; t < window.t1; ++t) {
                    for (int64_t h = window.h0; h < window.h1; ++h) {
                        for (int64_t w = window.w0; w < window.w1; ++w) {
                            pack.window_tgt_idx.push_back((t * height + h) * width + w);
                            pack.concat_tgt_idx.push_back(window_order_video_index++);
                            AppendRope(pack.freqs, kContextTokens + (t - window.t0), h - window.h0, w - window.w0);
                        }
                    }
                }
                for (int64_t text = 0; text < kContextTokens; ++text) {
                    pack.concat_tgt_idx.push_back(patched_tokens + text);
                    AppendRope(pack.freqs, text, text, text);
                }
            }
            if (window_order_video_index != patched_tokens) {
                throw std::runtime_error("SeedVR window metadata did not cover each patched token exactly once");
            }
            pack.window_src_idx.resize(static_cast<size_t>(patched_tokens));
            for (int64_t index = 0; index < patched_tokens; ++index) {
                pack.window_src_idx[static_cast<size_t>(pack.window_tgt_idx[static_cast<size_t>(index)])] = index;
            }

            pack.concat_src_idx.resize(pack.concat_tgt_idx.size());
            std::iota(pack.concat_src_idx.begin(), pack.concat_src_idx.end(), int64_t{0});
            std::stable_sort(pack.concat_src_idx.begin(), pack.concat_src_idx.end(),
                             [&](int64_t lhs, int64_t rhs) {
                                 return pack.concat_tgt_idx[static_cast<size_t>(lhs)] <
                                        pack.concat_tgt_idx[static_cast<size_t>(rhs)];
                             });

            int64_t max_length = 0;
            for (size_t window = 0; window < pack.WindowCount(); ++window) {
                const size_t base = window * 3;
                const int64_t length = pack.window_shape[base] * pack.window_shape[base + 1] *
                                       pack.window_shape[base + 2] + kContextTokens;
                max_length = std::max(max_length, length);
            }
            int64_t packed_offset = 0;
            for (size_t window = 0; window < pack.WindowCount(); ++window) {
                const size_t base = window * 3;
                const int64_t length = pack.window_shape[base] * pack.window_shape[base + 1] *
                                       pack.window_shape[base + 2] + kContextTokens;
                for (int64_t position = 0; position < max_length; ++position) {
                    const bool valid = position < length;
                    pack.padded_gather_idx.push_back(valid ? packed_offset + position : 0);
                    pack.valid_keys.push_back(static_cast<uint8_t>(valid));
                    if (valid) {
                        pack.padded_flat_idx.push_back(static_cast<int64_t>(window) * max_length + position);
                    }
                }
                packed_offset += length;
            }
            if (packed_offset != static_cast<int64_t>(pack.concat_tgt_idx.size())) {
                throw std::runtime_error("SeedVR padded attention metadata has an invalid token count");
            }
            return pack;
        }

        std::string ShapeString(std::initializer_list<int64_t> dimensions) {
            std::string result;
            for (const int64_t dimension: dimensions) {
                if (!result.empty()) {
                    result += 'x';
                }
                result += std::to_string(dimension);
            }
            return result;
        }

        struct MetadataDimensions {
            int64_t video_tokens;
            int64_t attention_tokens;
            int64_t padded_tokens;
            int64_t window_count;
            int64_t max_window_length;
        };

        MetadataDimensions ProfileMetadata(int64_t image_height, int64_t image_width, bool shifted, int64_t temporal = 1) {
            const int64_t patch_height = image_height / (8 * kPatchSize);
            const int64_t patch_width = image_width / (8 * kPatchSize);
            const auto windows = MakeWindows(patch_height, patch_width, shifted, temporal);
            int64_t max_window_length = 0;
            for (const auto &window: windows) {
                const int64_t length = (window.t1 - window.t0) * (window.h1 - window.h0) *
                                       (window.w1 - window.w0) + kContextTokens;
                max_window_length = std::max(max_window_length, length);
            }
            const int64_t window_count = static_cast<int64_t>(windows.size());
            return {
                temporal * patch_height * patch_width,
                temporal * patch_height * patch_width + window_count * kContextTokens,
                window_count * max_window_length,
                window_count,
                max_window_length
            };
        }

        std::string DitProfile(int64_t image_height, int64_t image_width,
                               const MetadataDimensions &normal, const MetadataDimensions &shifted, int64_t temporal = 1) {
            const auto pack = [](std::string_view prefix, const MetadataDimensions &metadata) {
                return std::string{prefix} +
                       "_window_tgt_idx:" + ShapeString({metadata.video_tokens}) + "," +
                       std::string{prefix} + "_window_src_idx:" +
                       ShapeString({metadata.video_tokens}) + "," +
                       std::string{prefix} + "_concat_tgt_idx:" + ShapeString({metadata.attention_tokens}) + "," +
                       std::string{prefix} + "_concat_src_idx:" + ShapeString({metadata.attention_tokens}) + "," +
                       std::string{
                           prefix
                       } + "_padded_flat_idx:" + ShapeString({metadata.attention_tokens}) + "," + std::string{prefix} +
                       "_padded_gather_idx:" + ShapeString({
                           metadata.padded_tokens
                       }) + "," + std::string{prefix} + "_valid_keys:" + ShapeString({
                           metadata.window_count, metadata.max_window_length
                       }) + "," + std::string{prefix} +
                       "_freqs:" + ShapeString({metadata.attention_tokens, kRopePairs, 2, 2});
            };
            const int64_t latent_height = image_height / 8;
            const int64_t latent_width = image_width / 8;
            return "latent:" + ShapeString({1, kLatentChannels, temporal, latent_height, latent_width}) +
                   ",timestep:1,context:" + ShapeString({1, kContextTokens, kContextWidth}) +
                   ",condition:" + ShapeString({1, kLatentChannels + 1, temporal, latent_height, latent_width}) + "," +
                   pack("normal", normal) + "," + pack("shifted", shifted);
        }

        common::ModelProfile ExactProfile(std::string shapes, std::string cache_subpath) {
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

        void RequireFile(const fs::path &path) {
            if (!fs::is_regular_file(path)) {
                throw std::runtime_error("Missing SeedVR runtime file: " + path.string());
            }
        }

        std::vector<Fp16> LoadContext(const fs::path &path) {
            RequireFile(path);
            constexpr size_t count = static_cast<size_t>(kContextTokens * kContextWidth);
            if (fs::file_size(path) != count * sizeof(Fp16)) {
                throw std::runtime_error("SeedVR positive context has unexpected size: " + path.string());
            }
            std::vector<Fp16> result(count);
            std::ifstream stream(path, std::ios::binary);
            stream.read(reinterpret_cast<char *>(result.data()),
                        static_cast<std::streamsize>(result.size() * sizeof(Fp16)));
            if (!stream) {
                throw std::runtime_error("Failed to read SeedVR positive context: " + path.string());
            }
            return result;
        }

        template<typename T>
        void FillBuffer(common::TensorBuffer<T> &buffer, const std::vector<T> &values) {
            std::copy(values.begin(), values.end(), buffer.HostData());
            buffer.CopyAsyncToDevice();
        }

        void BindMetadata(Ort::IoBinding &binding, const char *prefix, MetadataPack &pack,
                          common::TensorBuffer<int64_t> &window_tgt_idx, common::TensorBuffer<int64_t> &window_src_idx,
                          common::TensorBuffer<int64_t> &concat_tgt_idx, common::TensorBuffer<int64_t> &concat_src_idx,
                          common::TensorBuffer<int64_t> &padded_flat_idx,
                          common::TensorBuffer<int64_t> &padded_gather_idx,
                          common::TensorBuffer<bool> &valid_keys,
                          common::TensorBuffer<Fp16> &freqs) {
            const std::string root(prefix);
            binding.BindInput((root + "_window_tgt_idx").c_str(), window_tgt_idx.BindingValue());
            binding.BindInput((root + "_window_src_idx").c_str(), window_src_idx.BindingValue());
            binding.BindInput((root + "_concat_tgt_idx").c_str(), concat_tgt_idx.BindingValue());
            binding.BindInput((root + "_concat_src_idx").c_str(), concat_src_idx.BindingValue());
            binding.BindInput((root + "_padded_flat_idx").c_str(), padded_flat_idx.BindingValue());
            binding.BindInput((root + "_padded_gather_idx").c_str(), padded_gather_idx.BindingValue());
            binding.BindInput((root + "_valid_keys").c_str(), valid_keys.BindingValue());
            binding.BindInput((root + "_freqs").c_str(), freqs.BindingValue());
            FillBuffer(window_tgt_idx, pack.window_tgt_idx);
            FillBuffer(window_src_idx, pack.window_src_idx);
            FillBuffer(concat_tgt_idx, pack.concat_tgt_idx);
            FillBuffer(concat_src_idx, pack.concat_src_idx);
            FillBuffer(padded_flat_idx, pack.padded_flat_idx);
            FillBuffer(padded_gather_idx, pack.padded_gather_idx);
            for (size_t index = 0; index < pack.valid_keys.size(); ++index) {
                valid_keys.HostData()[index] = pack.valid_keys[index] != 0;
            }
            valid_keys.CopyAsyncToDevice();
            FillBuffer(freqs, pack.freqs);
        }
        // Scratch is device-only: full-resolution float accumulators do not need pinned host mirrors.
        void RunVaeTiles(common::OrtRunner &runner, Ort::RunOptions &options, cudaStream_t stream,
                         Ort::Value &input, Ort::Value &output, size_t height, size_t width,
                         size_t tile_height, size_t tile_width, size_t overlap, bool encode, size_t frames = 1) {
            const size_t latent_frames = (frames - 1) / 4 + 1;
            const size_t in_time = encode ? frames : latent_frames, out_time = encode ? latent_frames : frames;
            const size_t in_channels = (encode ? 3 : kLatentChannels) * in_time;
            const size_t out_channels = (encode ? kLatentChannels : 3) * out_time;
            const size_t numerator = encode ? 1 : 8, denominator = encode ? 8 : 1;
            const size_t out_height = height * numerator / denominator;
            const size_t out_width = width * numerator / denominator;
            const auto tiles = MakeVaeTiles(height, width, tile_height, tile_width, overlap);
            const auto tiles_x = std::count_if(tiles.begin(), tiles.end(), [](const SeedVrTile &tile) {
                return tile.y == 0;
            });
            const auto tiles_y = std::count_if(tiles.begin(), tiles.end(), [](const SeedVrTile &tile) {
                return tile.x == 0;
            });
            const size_t image_scale = encode ? 1 : 8;
            std::cout << "  Tile grid: X=" << tiles_x << ", Y=" << tiles_y
                      << " (" << tiles.size() << " tiles)\n"
                      << "  Tile size: X=" << tile_width * image_scale
                      << ", Y=" << tile_height * image_scale << " upscaled image pixels\n"
                      << "  Requested overlap: X=" << overlap * image_scale
                      << ", Y=" << overlap * image_scale << " upscaled image pixels" << std::endl;
            if (tiles.size() == 1) {
                Ort::IoBinding binding(runner.session);
                binding.BindInput(encode ? "image" : "latent", input);
                binding.BindOutput(encode ? "latent" : "image", output);
                try {
                    runner.session.Run(options, binding);
                    SEEDVR_CUDA_CHECK(cudaStreamSynchronize(stream));
                } catch (...) {
                    cudaStreamSynchronize(stream);
                    throw;
                }
                return;
            }
            std::vector<int64_t> in_shape{1, static_cast<int64_t>(in_channels),
                static_cast<int64_t>(tile_height), static_cast<int64_t>(tile_width)};
            std::vector<int64_t> out_shape{1, static_cast<int64_t>(out_channels),
                static_cast<int64_t>(tile_height * numerator / denominator),
                static_cast<int64_t>(tile_width * numerator / denominator)};
            if (frames > 1) {
                in_shape[1] /= in_time;
                in_shape.insert(in_shape.begin() + 2, static_cast<int64_t>(in_time));
                out_shape[1] /= out_time;
                out_shape.insert(out_shape.begin() + 2, static_cast<int64_t>(out_time));
            }
            const std::vector<int64_t> sum_shape{static_cast<int64_t>(out_channels * out_height * out_width)};
            const std::vector<int64_t> weight_shape{static_cast<int64_t>(out_height * out_width)};
            auto allocator = runner.DeviceAllocator();
            auto tile_in = Ort::Value::CreateTensor<Fp16>(allocator, in_shape.data(), in_shape.size());
            auto tile_out = Ort::Value::CreateTensor<Fp16>(allocator, out_shape.data(), out_shape.size());
            auto sums = Ort::Value::CreateTensor<float>(allocator, sum_shape.data(), sum_shape.size());
            auto weights = Ort::Value::CreateTensor<float>(allocator, weight_shape.data(), weight_shape.size());
            auto *sum = sums.GetTensorMutableData<float>();
            auto *weight = weights.GetTensorMutableData<float>();
            Ort::IoBinding binding(runner.session);
            binding.BindInput(encode ? "image" : "latent", tile_in);
            binding.BindOutput(encode ? "latent" : "image", tile_out);
            try {
                SEEDVR_CUDA_CHECK(cudaMemsetAsync(sum, 0, sum_shape[0] * sizeof(float), stream));
                SEEDVR_CUDA_CHECK(cudaMemsetAsync(weight, 0, weight_shape[0] * sizeof(float), stream));
                for (const auto &tile : tiles) {
                    LaunchSeedVrGatherTile(stream, input.GetTensorData<Fp16>(), tile_in.GetTensorMutableData<Fp16>(),
                                          in_channels, height, width, tile);
                    SEEDVR_CUDA_CHECK(cudaGetLastError());
                    runner.session.Run(options, binding);
                    LaunchSeedVrBlendTile(stream, tile_out.GetTensorData<Fp16>(), sum, weight,
                                          out_channels, out_height, out_width,
                                          ScaleVaeTile(tile, numerator, denominator));
                    SEEDVR_CUDA_CHECK(cudaGetLastError());
                }
                LaunchSeedVrNormalizeTiles(stream, sum, weight, output.GetTensorMutableData<Fp16>(),
                                           out_channels, out_height * out_width);
                SEEDVR_CUDA_CHECK(cudaGetLastError());
                SEEDVR_CUDA_CHECK(cudaStreamSynchronize(stream));
            } catch (...) {
                cudaStreamSynchronize(stream);
                throw;
            }
        }
    } // namespace

    class Pipeline::Impl {
    public:
        explicit Impl(Config config)
            : config_(std::move(config))
              , env_(ORT_LOGGING_LEVEL_WARNING, "SeedVR2") {
        }

        Image Run() {
            return Run(config_.input_path);
        }

        Image Run(const fs::path &input_path) {
            return RunClip({input_path}).front();
        }

        std::vector<Image> RunClip(const std::vector<fs::path> &input_paths) {
            const size_t frames = config_.video_frames;
            if (frames < 1 || (frames - 1) % 4 || input_paths.empty() || input_paths.size() > frames) {
                throw std::invalid_argument("SeedVR clip requires 1..video_frames images and video_frames=4n+1");
            }
            const bool video = frames > 1;
            if (video && config_.transformer_precision != "fp16") {
                throw std::invalid_argument("Export the video DiT in FP16; image quantized DiTs do not accept clips");
            }
            const int64_t latent_frames = static_cast<int64_t>((frames - 1) / 4 + 1);
            const auto &input_path = input_paths.front();
            if (config_.transformer_precision != "fp16" && config_.transformer_precision != "fp8_wo" &&
                config_.transformer_precision != "nvfp4") {
                throw std::invalid_argument("Transformer precision must be fp16, fp8_wo, or nvfp4");
            }
            ValidateVaeTiling(config_.vae_encoder_tile_size, config_.vae_encoder_overlap);
            ValidateVaeTiling(config_.vae_decoder_tile_size, config_.vae_decoder_overlap);
            if (config_.scale <= 0.0f) {
                throw std::invalid_argument("SeedVR scale must be positive");
            }
            if (config_.color_correction != "lab" && config_.color_correction != "none") {
                throw std::invalid_argument("SeedVR color correction must be lab or none");
            }
            RequireFile(input_path);
            const RawImage source = LoadPng(input_path);
            const size_t resized_height =
                    std::max<size_t>(2, static_cast<size_t>(std::nearbyint(source.height * config_.scale)));
            const size_t resized_width =
                    std::max<size_t>(2, static_cast<size_t>(std::nearbyint(source.width * config_.scale)));
            const RawImage resized = ResizeBicubic(source, resized_height, resized_width);
            std::vector<RawImage> resized_clip{resized};
            for (size_t i = 1; i < input_paths.size(); ++i) {
                const RawImage frame = LoadPng(input_paths[i]);
                if (frame.height != source.height || frame.width != source.width) {
                    throw std::invalid_argument("All video frames must have the same dimensions");
                }
                resized_clip.push_back(ResizeBicubic(frame, resized_height, resized_width));
            }
            const size_t padded_height = (resized_height + 15) / 16 * 16;
            const size_t padded_width = (resized_width + 15) / 16 * 16;
            const int64_t latent_height = static_cast<int64_t>(padded_height / 8);
            const int64_t latent_width = static_cast<int64_t>(padded_width / 8);
            const int64_t encoder_height = config_.vae_encoder_tile_size
                ? std::min(padded_height, static_cast<size_t>(config_.vae_encoder_tile_size)) : padded_height;
            const int64_t encoder_width = config_.vae_encoder_tile_size
                ? std::min(padded_width, static_cast<size_t>(config_.vae_encoder_tile_size)) : padded_width;
            const int64_t decoder_height = config_.vae_decoder_tile_size
                ? std::min(padded_height, static_cast<size_t>(config_.vae_decoder_tile_size)) / 8 : latent_height;
            const int64_t decoder_width = config_.vae_decoder_tile_size
                ? std::min(padded_width, static_cast<size_t>(config_.vae_decoder_tile_size)) / 8 : latent_width;
            const bool portrait = padded_height > padded_width;
            const int64_t max_height = portrait ? kProfileMaxLongSide : kProfileMaxShortSide;
            const int64_t max_width = portrait ? kProfileMaxShortSide : kProfileMaxLongSide;
            if (padded_height < kProfileMinImageSize || padded_width < kProfileMinImageSize ||
                padded_height > static_cast<size_t>(max_height) ||
                padded_width > static_cast<size_t>(max_width)) {
                throw std::invalid_argument("SeedVR padded input must be at least 32x32 and fit within the " +
                                            std::to_string(max_width) + "x" + std::to_string(max_height) +
                                            " TensorRT RTX profile");
            }
            MetadataPack normal = MakeMetadata(latent_height, latent_width, false, latent_frames);
            MetadataPack shifted = MakeMetadata(latent_height, latent_width, true, latent_frames);
            std::cout << "Input " << source.width << 'x' << source.height << " -> resized " << resized_width << 'x'
                    << resized_height << " -> padded " << padded_width << 'x' << padded_height << "\n"
                    << "Latent " << latent_frames << 'x' << latent_height << 'x' << latent_width
                    << " (T,H,W), normal windows=" << normal.WindowCount()
                    << ", shifted windows=" << shifted.WindowCount() << std::endl;

            const fs::path context_path = config_.model_dir / "seedvr2_3b_fp16_positive_context.bin";
            const std::vector<Fp16> context_values = LoadContext(context_path);

            if (sessions_ && (sessions_->padded_height != padded_height || sessions_->padded_width != padded_width)) {
                throw std::invalid_argument("SeedVR session is fixed to the first image's padded shape " +
                                            std::to_string(sessions_->padded_width) + "x" +
                                            std::to_string(sessions_->padded_height) + "; requested " +
                                            std::to_string(padded_width) + "x" + std::to_string(padded_height));
            }
            if (!sessions_) {
                const auto session_start = std::chrono::steady_clock::now();
                const fs::path encoder_path = config_.model_dir / (video ? "seedvr2_video_vae_encoder.onnx" : "seedvr2_ema_vae_encoder.onnx");
                const fs::path dit_path = config_.model_dir /
                        (video ? "seedvr2_video_dit.onnx" : "seedvr2_3b_" + config_.transformer_precision + "_dit.onnx");
                std::cout << "Transformer precision: " << config_.transformer_precision
                          << " (" << dit_path.generic_string() << ")" << std::endl;
                const fs::path decoder_path = config_.model_dir / (video ? "seedvr2_video_vae_decoder.onnx" : "seedvr2_ema_vae_decoder.onnx");
                RequireFile(encoder_path);
                RequireFile(dit_path);
                RequireFile(decoder_path);
                std::cout << "Creating SeedVR sessions for padded " << padded_width << 'x' << padded_height <<
                        std::endl;
                const auto trt_device = common::RegisterTensorRTRTXProvider(env_);
                const int cuda_device = common::ChooseCudaDeviceOrdinal(trt_device, "SEEDVR_CUDA_DEVICE_ID");
                SEEDVR_CUDA_CHECK(cudaSetDevice(cuda_device));
                compute_stream_ = common::CreateTensorRTRTXComputeStream(env_);
                auto *stream = &compute_stream_;
                const common::EpContextOptions ep_context{.output_dir = config_.ep_context_dir.string()};
                const std::string video_prefix = video ? "video" + std::to_string(frames) + "_" : "";
                const std::string image_size = video_prefix + std::to_string(padded_width) + "x" +
                                               std::to_string(padded_height);
                const auto normal_profile = ProfileMetadata(static_cast<int64_t>(padded_height),
                                                            static_cast<int64_t>(padded_width), false, latent_frames);
                const auto shifted_profile = ProfileMetadata(static_cast<int64_t>(padded_height),
                                                             static_cast<int64_t>(padded_width), true, latent_frames);

                auto sessions = std::make_unique<SessionBundle>();
                sessions->padded_height = padded_height;
                sessions->padded_width = padded_width;
                // TODO: TensorRT RTX attention decomposition fails with dynamic spatial shapes;
                // restore ranged profiles when that path supports them.
                auto encoder_profile = ExactProfile("image:" + (video
                                                    ? ShapeString({1, 3, static_cast<int64_t>(frames), encoder_height, encoder_width})
                                                    : ShapeString({1, 3, encoder_height, encoder_width})),
                                                    video_prefix + "vae_encoder_" + std::to_string(encoder_width) + "x" +
                                                    std::to_string(encoder_height));
                encoder_profile.skip_compile = true;
                sessions->encoder = std::make_unique<common::OrtRunner>(
                    env_, encoder_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                    encoder_profile, stream);
                auto dit_profile = ExactProfile(
                    DitProfile(static_cast<int64_t>(padded_height), static_cast<int64_t>(padded_width),
                               normal_profile, shifted_profile, latent_frames),
                    "dit_" + config_.transformer_precision + "_" + image_size);
                // The exporter gives each dynamic ONNX input dimension a stable name.
                // Specializing them at session creation avoids a TensorRT RTX Myelin
                // shape-inference failure when the DiT uses the full resolution range.
                dit_profile.free_dimension_overrides = {
                    {"latent_dim3", latent_height}, {"latent_dim4", latent_width},
                    {"normal_window_tgt_idx_dim0", static_cast<int64_t>(normal.window_tgt_idx.size())},
                    {"normal_window_src_idx_dim0", static_cast<int64_t>(normal.window_src_idx.size())},
                    {"normal_concat_tgt_idx_dim0", static_cast<int64_t>(normal.concat_tgt_idx.size())},
                    {"normal_concat_src_idx_dim0", static_cast<int64_t>(normal.concat_src_idx.size())},
                    {"normal_padded_flat_idx_dim0", static_cast<int64_t>(normal.padded_flat_idx.size())},
                    {"normal_padded_gather_idx_dim0", static_cast<int64_t>(normal.padded_gather_idx.size())},
                    {"normal_valid_keys_dim0", static_cast<int64_t>(normal.WindowCount())},
                    {"normal_valid_keys_dim1", static_cast<int64_t>(normal.MaxWindowLength())},
                    {"shifted_window_tgt_idx_dim0", static_cast<int64_t>(shifted.window_tgt_idx.size())},
                    {"shifted_concat_tgt_idx_dim0", static_cast<int64_t>(shifted.concat_tgt_idx.size())},
                    {"shifted_concat_src_idx_dim0", static_cast<int64_t>(shifted.concat_src_idx.size())},
                    {"shifted_padded_flat_idx_dim0", static_cast<int64_t>(shifted.padded_flat_idx.size())},
                    {"shifted_padded_gather_idx_dim0", static_cast<int64_t>(shifted.padded_gather_idx.size())},
                    {"shifted_valid_keys_dim0", static_cast<int64_t>(shifted.WindowCount())},
                    {"shifted_valid_keys_dim1", static_cast<int64_t>(shifted.MaxWindowLength())}
                };
                if (!config_.weight_streaming_budget.empty()) {
                    dit_profile.extra_ep_options.emplace_back("nv_weight_streaming_budget",
                                                              config_.weight_streaming_budget);
                }
                sessions->dit = std::make_unique<common::OrtRunner>(
                    env_, dit_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context, dit_profile, stream);
                sessions->decoder = std::make_unique<common::OrtRunner>(
                    env_, decoder_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                    ExactProfile("latent:" + (video
                                 ? ShapeString({1, kLatentChannels, latent_frames, decoder_height, decoder_width})
                                 : ShapeString({1, kLatentChannels, decoder_height, decoder_width})),
                                 video_prefix + std::string("vae_decoder_unclamped_") +
                                 std::to_string(decoder_width * 8) + "x" + std::to_string(decoder_height * 8)),
                    stream);
                if (!sessions->encoder->HasDeviceIo() || !sessions->dit->HasDeviceIo() ||
                    !sessions->decoder->HasDeviceIo()) {
                    throw std::runtime_error("SeedVR requires CUDA-backed TensorRT RTX I/O");
                }
                sessions_ = std::move(sessions);
                const double session_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - session_start).count();
                std::cout << "SeedVR session creation: " << session_ms << " ms" << std::endl;
            }
            const auto inference_start = std::chrono::steady_clock::now();
            auto &encoder = *sessions_->encoder;
            auto &dit = *sessions_->dit;
            auto &decoder = *sessions_->decoder;

            const size_t spatial_size = static_cast<size_t>(latent_frames * latent_height * latent_width);
            std::vector<int64_t> image_shape{
                1, 3, static_cast<int64_t>(padded_height),
                static_cast<int64_t>(padded_width)
            };
            std::vector<int64_t> encoded_shape{1, kLatentChannels, latent_height, latent_width};
            const std::vector<int64_t> latent_shape{1, kLatentChannels, latent_frames, latent_height, latent_width};
            const std::vector<int64_t> condition_shape{1, kLatentChannels + 1, latent_frames, latent_height, latent_width};
            std::vector<int64_t> output_shape{
                1, 3, static_cast<int64_t>(padded_height),
                static_cast<int64_t>(padded_width)
            };

            const auto frame_shape = output_shape;
            if (video) {
                image_shape.insert(image_shape.begin() + 2, static_cast<int64_t>(frames));
                output_shape.insert(output_shape.begin() + 2, static_cast<int64_t>(frames));
                encoded_shape.insert(encoded_shape.begin() + 2, latent_frames);
            }
            const auto cuda_stream = reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
            // Drain queued work before buffers are destroyed, including exception paths.
            struct StreamDrain {
                cudaStream_t stream;
                ~StreamDrain() { cudaStreamSynchronize(stream); }
            };
            Ort::RunOptions run_options;
            run_options.SetSyncStream(compute_stream_);
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
            common::TensorBuffer<Fp16> sampled(decoder, encoded_shape, true);
            {
                common::TensorBuffer<Fp16> condition(dit, condition_shape, true);
                {
                    common::TensorBuffer<Fp16> image(encoder, image_shape, true);
                    common::TensorBuffer<Fp16> encoded(encoder, encoded_shape, true);
                    Ort::IoBinding encoder_io(encoder.session);
                    StreamDrain drain{cuda_stream};
                    const size_t image_plane = padded_height * padded_width;
                    std::vector<Fp16> clip_pixels(3 * frames * image_plane);
                    for (size_t t = 0; t < frames; ++t) {
                        const auto pixels = PadToNchwFp16(resized_clip[std::min(t, resized_clip.size() - 1)], padded_height, padded_width);
                        for (size_t c = 0; c < 3; ++c) {
                            std::copy_n(pixels.data() + c * image_plane, image_plane,
                                        clip_pixels.data() + (c * frames + t) * image_plane);
                        }
                    }
                    FillBuffer(image, clip_pixels);
                    std::cout << "Running VAE encoder" << std::endl;
                    if (config_.vae_encoder_tile_size) {
                        RunVaeTiles(encoder, run_options, cuda_stream, image.BindingValue(), encoded.BindingValue(),
                                    padded_height, padded_width, encoder_height, encoder_width,
                                    config_.vae_encoder_overlap, true, frames);
                    } else {
                        encoder_io.BindInput("image", image.BindingValue());
                        encoder_io.BindOutput("latent", encoded.BindingValue());
                        encoder.session.Run(run_options, encoder_io);
                    }
                    LaunchSeedVrConditionKernel(cuda_stream, encoded.BindingValue().GetTensorData<Fp16>(),
                                                condition.BindingValue().GetTensorMutableData<Fp16>(), spatial_size);
                    SEEDVR_CUDA_CHECK(cudaGetLastError());
                    SEEDVR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
                }
                common::TensorBuffer<Fp16> latent(dit, latent_shape, true);
                common::TensorBuffer<Fp16> timestep(dit, {1}, true);
                common::TensorBuffer<Fp16> context(dit, {1, kContextTokens, kContextWidth}, true);
                common::TensorBuffer<Fp16> velocity(dit, latent_shape, true);

                auto make_pack_buffers = [&](common::OrtRunner &runner, const MetadataPack &pack) {
                    struct Buffers {
                        common::TensorBuffer<int64_t> window_tgt_idx;
                        common::TensorBuffer<int64_t> window_src_idx;
                        common::TensorBuffer<int64_t> concat_tgt_idx;
                        common::TensorBuffer<int64_t> concat_src_idx;
                        common::TensorBuffer<int64_t> padded_flat_idx;
                        common::TensorBuffer<int64_t> padded_gather_idx;
                        common::TensorBuffer<bool> valid_keys;
                        common::TensorBuffer<Fp16> freqs;
                    };
                    const int64_t attention = static_cast<int64_t>(pack.concat_tgt_idx.size());
                    return Buffers{
                        common::TensorBuffer<int64_t>(runner, {static_cast<int64_t>(pack.window_tgt_idx.size())}, true),
                        common::TensorBuffer<int64_t>(runner, {static_cast<int64_t>(pack.window_src_idx.size())}, true),
                        common::TensorBuffer<int64_t>(runner, {attention}, true),
                        common::TensorBuffer<int64_t>(runner, {attention}, true),
                        common::TensorBuffer<int64_t>(runner, {attention}, true),
                        common::TensorBuffer<int64_t>(runner,
                                                      {static_cast<int64_t>(pack.padded_gather_idx.size())}, true),
                        common::TensorBuffer<bool>(runner, {
                                                       static_cast<int64_t>(pack.WindowCount()),
                                                       static_cast<int64_t>(pack.MaxWindowLength())
                                                   }, true),
                        common::TensorBuffer<Fp16>(runner, {attention, kRopePairs, 2, 2}, true)
                    };
                };
                auto normal_buffers = make_pack_buffers(dit, normal);
                auto shifted_buffers = make_pack_buffers(dit, shifted);

                Ort::IoBinding dit_io(dit.session);
                std::unique_ptr<common::TensorBuffer<float> > cuda_noise;
                StreamDrain drain{cuda_stream};
                dit_io.BindInput("latent", latent.BindingValue());
                dit_io.BindInput("timestep", timestep.BindingValue());
                dit_io.BindInput("context", context.BindingValue());
                dit_io.BindInput("condition", condition.BindingValue());
                dit_io.BindOutput("denoised_latent", velocity.BindingValue());
                BindMetadata(dit_io, "normal", normal, normal_buffers.window_tgt_idx, normal_buffers.window_src_idx,
                             normal_buffers.concat_tgt_idx, normal_buffers.concat_src_idx,
                             normal_buffers.padded_flat_idx, normal_buffers.padded_gather_idx,
                             normal_buffers.valid_keys, normal_buffers.freqs);
                BindMetadata(dit_io, "shifted", shifted, shifted_buffers.window_tgt_idx, shifted_buffers.window_src_idx,
                             shifted_buffers.concat_tgt_idx, shifted_buffers.concat_src_idx,
                             shifted_buffers.padded_flat_idx, shifted_buffers.padded_gather_idx,
                             shifted_buffers.valid_keys, shifted_buffers.freqs);
                FillBuffer(context, context_values);
                timestep.HostData()[0] = Fp16(1000.0f);
                timestep.CopyAsyncToDevice();
                const size_t noise_count = static_cast<size_t>(kLatentChannels) * spatial_size;
                if (config_.rng == common::Rng::CudaPhilox) {
                    cuda_noise = std::make_unique<common::TensorBuffer<float> >(
                        dit, std::vector<int64_t>{static_cast<int64_t>(noise_count)}, true);
                    common::GenerateNormalCuda(cuda_stream,
                                               cuda_noise->BindingValue().GetTensorMutableData<float>(),
                                               noise_count, config_.seed);
                    LaunchSeedVrFloatToHalfKernel(cuda_stream,
                                                  cuda_noise->BindingValue().GetTensorData<float>(),
                                                  latent.BindingValue().GetTensorMutableData<Fp16>(), noise_count);
                    SEEDVR_CUDA_CHECK(cudaGetLastError());
                } else {
                    std::vector<float> values(noise_count);
                    common::GenerateNormalCpu(values, config_.seed, config_.rng);
                    std::vector<Fp16> noise(noise_count);
                    std::transform(values.begin(), values.end(), noise.begin(),
                                   [](float value) { return Fp16(value); });
                    FillBuffer(latent, noise);
                }

                std::cout << "Running SeedVR2 DiT" << std::endl;
                dit.session.Run(run_options, dit_io);
                LaunchSeedVrEulerKernel(cuda_stream, latent.BindingValue().GetTensorData<Fp16>(),
                                        velocity.BindingValue().GetTensorData<Fp16>(),
                                        sampled.BindingValue().GetTensorMutableData<Fp16>(), noise_count);
                SEEDVR_CUDA_CHECK(cudaGetLastError());
                SEEDVR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
            }
            common::TensorBuffer<Fp16> decoded(decoder, output_shape, true);
            Ort::IoBinding decoder_io(decoder.session);
            StreamDrain drain{cuda_stream};
            std::cout << "Running VAE decoder" << std::endl;
            if (config_.vae_decoder_tile_size) {
                RunVaeTiles(decoder, run_options, cuda_stream, sampled.BindingValue(), decoded.BindingValue(),
                            latent_height, latent_width, decoder_height, decoder_width,
                            config_.vae_decoder_overlap / 8, false, frames);
            } else {
                decoder_io.BindInput("latent", sampled.BindingValue());
                decoder_io.BindOutput("image", decoded.BindingValue());
                decoder.session.Run(run_options, decoder_io);
            }
            LaunchSeedVrClampKernel(cuda_stream, decoded.BindingValue().GetTensorMutableData<Fp16>(),
                                    3 * frames * padded_height * padded_width);
            SEEDVR_CUDA_CHECK(cudaGetLastError());
            std::vector<Image> results;
            common::TensorBuffer<Fp16> output_frame(decoder, frame_shape, true);
            const size_t padded_plane = padded_height * padded_width;
            for (size_t t = 0; t < input_paths.size(); ++t) {
                for (size_t c = 0; c < 3; ++c) {
                    SEEDVR_CUDA_CHECK(cudaMemcpyAsync(
                        output_frame.BindingValue().GetTensorMutableData<Fp16>() + c * padded_plane,
                        decoded.BindingValue().GetTensorData<Fp16>() + (c * frames + t) * padded_plane,
                        padded_plane * sizeof(Fp16), cudaMemcpyDeviceToDevice, cuda_stream));
                }
                if (config_.color_correction == "lab") {
                    common::TensorBuffer<float> reference(decoder, frame_shape, true);
                    FillBuffer(reference, PadToNchwFloat(resized_clip[t], padded_height, padded_width));
                    LaunchSeedVrLabKernel(cuda_stream, output_frame.BindingValue().GetTensorMutableData<Fp16>(),
                                          reference.BindingValue().GetTensorData<float>(),
                                          resized_height, resized_width, padded_width, padded_plane);
                    SEEDVR_CUDA_CHECK(cudaGetLastError());
                    SEEDVR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
                }
                auto output_ready = output_frame.CopyAsyncToHostWithNotification();
                output_ready.Sync();

                Image result;
                result.height = resized_height - resized_height % 2;
                result.width = resized_width - resized_width % 2;
                result.chw.resize(3 * result.height * result.width);
                const Fp16 *decoded_data = output_frame.HostData();
                const size_t result_plane = result.height * result.width;
                for (size_t channel = 0; channel < 3; ++channel) {
                    for (size_t y = 0; y < result.height; ++y) {
                        for (size_t x = 0; x < result.width; ++x) {
                            result.chw[channel * result_plane + y * result.width + x] =
                                    decoded_data[channel * padded_plane + y * padded_width + x].ToFloat();
                        }
                    }
                }
                results.push_back(std::move(result));
            }
            // Include completed GPU work and output transfers in the wall-clock timing.
            SEEDVR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
            const double inference_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - inference_start).count();
            std::cout << "SeedVR inference: " << inference_ms << " ms total, "
                      << inference_ms / static_cast<double>(input_paths.size()) << " ms/image ("
                      << input_paths.size() << " images)" << std::endl;
            return results;
        }

    private:
        struct SessionBundle {
            size_t padded_height = 0;
            size_t padded_width = 0;
            std::unique_ptr<common::OrtRunner> encoder;
            std::unique_ptr<common::OrtRunner> dit;
            std::unique_ptr<common::OrtRunner> decoder;
        };

        Config config_;
        Ort::Env env_;
        Ort::SyncStream compute_stream_{nullptr};
        std::unique_ptr<SessionBundle> sessions_;
    };

    Pipeline::Pipeline(Config config)
        : impl_(std::make_unique<Impl>(std::move(config))) {
    }

    Pipeline::~Pipeline() = default;

    Image Pipeline::Run() {
        return impl_->Run();
    }

    Image Pipeline::Run(const std::filesystem::path &input_path) {
        return impl_->Run(input_path);
    }
    std::vector<Image> Pipeline::RunClip(const std::vector<std::filesystem::path> &input_paths) {
        return impl_->RunClip(input_paths);
    }
} // namespace din::seedvr
