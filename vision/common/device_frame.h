// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>

namespace din::vision
{
    // Borrowed device pointer at the bottom-left pixel, signed byte stride, float RGBA.
    struct DeviceRgbaView
    {
        void* data;
        std::ptrdiff_t row_bytes;
        int width, height;
        bool premultiplied;
    };

    // Output points at the first pixel of a window relative to the source bottom-left.
    struct DeviceOutputView
    {
        void* data;
        std::ptrdiff_t row_bytes;
        int x, y, width, height;
        bool masked_alpha;
    };
}
