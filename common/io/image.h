// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace din::io
{

// Top-down, tightly packed RGB8 pixels at the source image's dimensions.
struct RgbImage
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;
};

enum class ImageValueRange
{
    ZeroTo255,
    MinusOneToOne,
};

// PNG helpers throw on failure. Writers create missing parent directories.
// Byte buffers are tightly packed, interleaved RGB or single-channel gray.
RgbImage LoadPng(const std::filesystem::path& path);
void SaveRgbPng(const std::filesystem::path& path, const uint8_t* rgb, uint32_t width, uint32_t height);
void SaveGrayPng(const std::filesystem::path& path, const uint8_t* gray, uint32_t width, uint32_t height);

// Float buffers are planar CHW (RGB or one gray channel). The explicit range
// maps to [0,255]; out-of-range values are clamped, then rounded to the nearest
// byte. Non-finite values are rejected. Loading always returns RGB8.
void SaveRgbPng(const std::filesystem::path& path, const float* chw_data, uint32_t width, uint32_t height,
                ImageValueRange range);
void SaveGrayPng(const std::filesystem::path& path, const float* gray, uint32_t width, uint32_t height,
                 ImageValueRange range);
}  // namespace din::io
