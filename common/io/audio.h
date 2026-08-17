// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
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

Audio LoadAudio(const std::string& path, int target_rate);
Audio LoadWavMono(const std::string& path);

}  // namespace din::io
