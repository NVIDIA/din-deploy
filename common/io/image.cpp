// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "lodepng.h"

namespace din::io
{
namespace
{

size_t ImageSize(uint32_t width, uint32_t height, size_t channels)
{
    if (!width || !height || size_t(width) > std::numeric_limits<size_t>::max() / height / channels)
        throw std::invalid_argument("Invalid PNG dimensions");
    return size_t(width) * height * channels;
}

void SavePng(const std::filesystem::path& path, const uint8_t* pixels, uint32_t width, uint32_t height,
             LodePNGColorType color)
{
    if (!pixels)
        throw std::invalid_argument("Invalid PNG pixels");
    ImageSize(width, height, color == LCT_RGB ? 3 : 1);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    const unsigned error = lodepng::encode(path.string(), pixels, width, height, color, 8);
    if (error)
        throw std::runtime_error("failed to save PNG " + path.string() + ": " + lodepng_error_text(error));
}

uint8_t FloatToByte(float value, ImageValueRange range)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Non-finite PNG pixel value");
    switch (range)
    {
    case ImageValueRange::ZeroTo255:
        break;
    case ImageValueRange::MinusOneToOne:
        value = value * 127.5f + 127.5f;
        break;
    default:
        throw std::invalid_argument("Invalid PNG image value range");
    }
    return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
}

std::vector<uint8_t> ToBytes(const float* chw_data, uint32_t width, uint32_t height, size_t channels,
                             ImageValueRange range)
{
    if (!chw_data)
        throw std::invalid_argument("Invalid PNG pixels");
    std::vector<uint8_t> pixels(ImageSize(width, height, channels));
    const size_t plane = size_t(width) * height;
    for (size_t p = 0; p < plane; ++p)
        for (size_t c = 0; c < channels; ++c)
            pixels[p * channels + c] = FloatToByte(chw_data[c * plane + p], range);
    return pixels;
}

}  // namespace

RgbImage LoadPng(const std::filesystem::path& path)
{
    RgbImage image;
    const unsigned error = lodepng::decode(image.rgb, image.width, image.height, path.string(), LCT_RGB, 8);
    if (error)
        throw std::runtime_error("failed to load PNG " + path.string() + ": " + lodepng_error_text(error));
    return image;
}

RgbImageView LoadPng(const std::filesystem::path& path, std::span<uint8_t> destination)
{
    std::vector<uint8_t> compressed, native;
    lodepng::State state;
    state.decoder.color_convert = 0;
    uint32_t width = 0, height = 0;
    unsigned error = lodepng::load_file(compressed, path.string());
    if (!error)
        error = lodepng_inspect(&width, &height, &state, compressed.data(), compressed.size());
    if (!error && ImageSize(width, height, 3) > destination.size())
        throw std::invalid_argument("PNG exceeds destination capacity: " + path.string());
    if (!error)
        error = lodepng::decode(native, width, height, state, compressed);
    if (!error)
    {
        lodepng::State rgb;
        rgb.info_raw.colortype = LCT_RGB;
        rgb.info_raw.bitdepth = 8;
        error = lodepng_convert(destination.data(), native.data(), &rgb.info_raw,
                                &state.info_png.color, width, height);
    }
    if (error)
        throw std::runtime_error("failed to load PNG " + path.string() + ": " + lodepng_error_text(error));
    return {width, height, destination.first(ImageSize(width, height, 3))};
}

std::pair<uint32_t, uint32_t> ReadPngSize(const std::filesystem::path& path)
{
    std::array<uint8_t, 33> header{};
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    lodepng::State state;
    uint32_t width = 0, height = 0;
    const unsigned error = lodepng_inspect(&width, &height, &state, header.data(), size_t(file.gcount()));
    if (error)
        throw std::runtime_error("failed to read PNG size " + path.string() + ": " + lodepng_error_text(error));
    ImageSize(width, height, 3);
    return {width, height};
}

void SaveRgbPng(const std::filesystem::path& path, const uint8_t* rgb, uint32_t width, uint32_t height)
{
    SavePng(path, rgb, width, height, LCT_RGB);
}

void SaveGrayPng(const std::filesystem::path& path, const uint8_t* gray, uint32_t width, uint32_t height)
{
    SavePng(path, gray, width, height, LCT_GREY);
}

void SaveRgbPng(const std::filesystem::path& path, const float* chw_data, uint32_t width, uint32_t height,
                ImageValueRange range)
{
    const auto pixels = ToBytes(chw_data, width, height, 3, range);
    SaveRgbPng(path, pixels.data(), width, height);
}

void SaveGrayPng(const std::filesystem::path& path, const float* gray, uint32_t width, uint32_t height,
                 ImageValueRange range)
{
    const auto pixels = ToBytes(gray, width, height, 1, range);
    SaveGrayPng(path, pixels.data(), width, height);
}

}  // namespace din::io
