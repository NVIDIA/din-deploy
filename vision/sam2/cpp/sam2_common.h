// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace din::sam2
{

struct PromptOptimizationShapes
{
    std::vector<int64_t> point_coords{1, 16, 2};
    std::vector<int64_t> point_labels{1, 16};
    std::vector<int64_t> input_masks{1, 1, 256, 256};
    std::vector<int64_t> has_input_masks{1, 1, 1, 1};
};

struct DecoderOptimizationShapes
{
    std::vector<int64_t> image_features_0{1, 32, 256, 256};
    std::vector<int64_t> image_features_1{1, 64, 128, 128};
};

struct Sam2Config
{
    int64_t image_size = 1024;
    int64_t max_points = 16;
    int64_t max_memory_frames = 7;
    int64_t hidden_dim = 256;
    int64_t memory_dim = 64;
    int64_t max_input_size = 4096;
    std::string dtype = "float16";
    std::string spatial_contract = "static_1024";
    PromptOptimizationShapes prompt_optimization_shapes;
    DecoderOptimizationShapes decoder_optimization_shapes;

    [[nodiscard]] int64_t embedding_size() const
    {
        return image_size / 16;
    }

    [[nodiscard]] int64_t low_res_mask_size() const
    {
        return image_size / 4;
    }

    [[nodiscard]] int64_t tokens() const
    {
        const int64_t size = embedding_size();
        return size * size;
    }
};

struct Image
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;
};

struct Prompt
{
    std::vector<float> coords;
    std::vector<int32_t> labels;
};

}  // namespace din::sam2
