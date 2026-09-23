// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define MINIAUDIO_IMPLEMENTATION
#include "audio.h"

#include <stdexcept>

#include <miniaudio.h>

namespace din::io
{
Audio LoadAudio(const std::filesystem::path& path, int target_rate)
{
    if (target_rate <= 0)
        throw std::invalid_argument("Audio sample rate must be positive");
    auto check = [](ma_result result)
    {
        if (result != MA_SUCCESS)
            throw std::runtime_error(std::string("Decode audio: ") + ma_result_description(result));
    };
    ma_decoder decoder;
    auto config = ma_decoder_config_init(ma_format_f32, 1, target_rate);
#ifdef _WIN32
    check(ma_decoder_init_file_w(path.c_str(), &config, &decoder));
#else
    check(ma_decoder_init_file(path.c_str(), &config, &decoder));
#endif
    struct Guard
    {
        ma_decoder* decoder;
        ~Guard()
        {
            ma_decoder_uninit(decoder);
        }
    } guard{&decoder};
    Audio audio;
    audio.sample_rate = target_rate;
    float block[16384];
    for (;;)
    {
        ma_uint64 count = 0;
        const auto status = ma_decoder_read_pcm_frames(&decoder, block, 16384, &count);
        if (status != MA_SUCCESS && status != MA_AT_END)
            check(status);
        audio.samples.insert(audio.samples.end(), block, block + count);
        if (status == MA_AT_END || count == 0)
            break;
    }
    if (audio.samples.empty())
        throw std::runtime_error("The audio file contains no samples");
    return audio;
}
}  // namespace din::io
