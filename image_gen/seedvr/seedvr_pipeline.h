// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "rng.h"

namespace din::seedvr {
    struct Config {
        std::filesystem::path model_dir = "models/SeedVR2-3B-onnx";
        std::string transformer_precision = "fp16";
        std::filesystem::path input_path;
        std::filesystem::path output_path = "out/seedvr/seedvr-cpp.png";
        std::filesystem::path ep_cache_dir = "out/seedvr/trt_rtx_cache_cpp";
        std::filesystem::path ep_context_dir = "out/seedvr/ep_context_cpp";
        float scale = 2.0f;
        uint32_t seed = 0;
        common::Rng rng = common::Rng::CpuStd;
        std::string weight_streaming_budget;
        std::string color_correction = "lab";
        // Tile and overlap dimensions in upscaled image pixels; zero size disables tiling.
        int vae_encoder_tile_size = 0;
        int vae_encoder_overlap = 64;
        int vae_decoder_tile_size = 0;
        int vae_decoder_overlap = 64;
    };

    struct Image {
        std::vector<float> chw;
        size_t height = 0;
        size_t width = 0;
    };

    class Pipeline {
    public:
        explicit Pipeline(Config config);

        ~Pipeline();

        Pipeline(const Pipeline &) = delete;

        Pipeline &operator=(const Pipeline &) = delete;

        Image Run();

        Image Run(const std::filesystem::path &input_path);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace din::seedvr
