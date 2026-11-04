// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include "../../common/device_frame.h"

namespace din::sam2 {
// Device pointer at the bottom-left pixel; signed byte stride, float RGBA.
using DeviceRgbaView = din::vision::DeviceRgbaView;
// Window coordinates are relative to the input bottom-left; output points to its first pixel.
using DeviceOutputView = din::vision::DeviceOutputView;
void ConvertDeviceRgba(const DeviceRgbaView& source, uint8_t* rgb, void* stream);
void WriteDeviceMatte(const DeviceRgbaView& source, const DeviceOutputView& output,
                      const void* half_logits, int mask_size, void* stream);
}
