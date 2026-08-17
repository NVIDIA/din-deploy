// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

namespace din::io
{

inline constexpr unsigned int image_dim = 224;

enum class ImageValueRange
{
    ZeroTo255,
    MinusOneToOne,
};

void loadInputImage(float* pData, const char* imageFileName);
void saveOutputImage(float* pData, const char* imageFileName);

bool SaveRgbFloatImage(const std::string& path, const float* chw_data, size_t height, size_t width,
                       ImageValueRange range);

}  // namespace din::io
