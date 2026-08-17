// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lodepng.h"

namespace din::io
{
namespace
{

static unsigned char clampAndConvert(float val)
{
    if (val < 0)
        val = 0;
    if (val > 255)
        val = 255;
    return static_cast<unsigned char>(val);
}

unsigned char floatToByte(float value, ImageValueRange range)
{
    if (range == ImageValueRange::MinusOneToOne)
    {
        value = value * 127.5f + 127.5f;
    }

    return static_cast<unsigned char>(std::lround(std::clamp(value, 0.0f, 255.0f)));
}

}  // namespace

void loadInputImage(float* pData, const char* imageFileName)
{
    unsigned char* image = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int error = lodepng_decode32_file(&image, &width, &height, imageFileName);
    if (error)
    {
        std::printf("\nFailed to load the input image: %s. Exiting\n", lodepng_error_text(error));
        std::exit(EXIT_FAILURE);
    }

    if (width != image_dim || height != image_dim)
    {
        std::printf("\nImage not of right size (%ux%u, expected %ux%u). Exiting\n", width, height, image_dim,
                    image_dim);
        std::exit(EXIT_FAILURE);
    }

    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            unsigned char r = image[(y * width + x) * 4 + 0];
            unsigned char g = image[(y * width + x) * 4 + 1];
            unsigned char b = image[(y * width + x) * 4 + 2];

            pData[0 * width * height + y * width + x] = static_cast<float>(b);
            pData[1 * width * height + y * width + x] = static_cast<float>(g);
            pData[2 * width * height + y * width + x] = static_cast<float>(r);
        }
    }

    std::free(image);
}

void saveOutputImage(float* pData, const char* imageFileName)
{
    unsigned int width = image_dim;
    unsigned int height = image_dim;

    std::vector<unsigned char> image(width * height * 4);
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            float b = pData[0 * width * height + y * width + x];
            float g = pData[1 * width * height + y * width + x];
            float r = pData[2 * width * height + y * width + x];

            image[(y * width + x) * 4 + 0] = clampAndConvert(r);
            image[(y * width + x) * 4 + 1] = clampAndConvert(g);
            image[(y * width + x) * 4 + 2] = clampAndConvert(b);
            image[(y * width + x) * 4 + 3] = 255;
        }
    }
    auto encode_error = lodepng_encode32_file(imageFileName, image.data(), width, height);
    if (encode_error != 0)
    {
        std::printf("Failed to save output image \"%s\": %u\n", imageFileName, encode_error);
    }
}

bool SaveRgbFloatImage(const std::string& path, const float* chw_data, size_t height, size_t width,
                       ImageValueRange range)
{
    if (chw_data == nullptr || height == 0 || width == 0)
    {
        return false;
    }

    const size_t plane_size = height * width;
    std::vector<unsigned char> image(plane_size * 4);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            const size_t pixel = y * width + x;
            const size_t rgba = pixel * 4;
            image[rgba + 0] = floatToByte(chw_data[pixel], range);
            image[rgba + 1] = floatToByte(chw_data[plane_size + pixel], range);
            image[rgba + 2] = floatToByte(chw_data[2 * plane_size + pixel], range);
            image[rgba + 3] = 255;
        }
    }

    const unsigned int error = lodepng_encode32_file(path.c_str(), image.data(), static_cast<unsigned int>(width),
                                                     static_cast<unsigned int>(height));
    if (error != 0)
    {
        std::printf("Failed to save image to %s: %s\n", path.c_str(), lodepng_error_text(error));
        return false;
    }

    return true;
}

}  // namespace din::io
