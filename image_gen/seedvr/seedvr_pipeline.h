// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace din::seedvr
{
    struct Config
    {
        std::filesystem::path model_dir = "out/seedvr/onnx";
        std::filesystem::path input_path;
        std::filesystem::path output_path = "out/seedvr/seedvr-cpp.png";
        std::filesystem::path ep_cache_dir = "out/seedvr/trt_rtx_cache_cpp";
        std::filesystem::path ep_context_dir = "out/seedvr/ep_context_cpp";
        float scale = 2.0f;
        uint32_t seed = 0;
    };

    struct Image
    {
        std::vector<float> chw;
        size_t height = 0;
        size_t width = 0;
    };

    class Pipeline
    {
    public:
        explicit Pipeline(Config config);
        ~Pipeline();
        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        Image Run();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace din::seedvr
