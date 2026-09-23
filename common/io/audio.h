// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

namespace din::io
{

struct Audio
{
    std::vector<float> samples;
    int sample_rate = 0;

    [[nodiscard]] double Duration() const
    {
        return sample_rate > 0 ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
};

// Decode to mono floats at the requested sample rate.
Audio LoadAudio(const std::filesystem::path& path, int target_rate);

}  // namespace din::io
