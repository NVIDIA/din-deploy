// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "audio.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

namespace din::io
{
namespace
{

uint32_t ReadU32(std::ifstream& stream)
{
    uint32_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint16_t ReadU16(std::ifstream& stream)
{
    uint16_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::string ReadTag(std::ifstream& stream)
{
    char tag[4];
    stream.read(tag, 4);
    return std::string(tag, 4);
}

std::string LowercaseExtension(const std::string& path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos)
    {
        return {};
    }
    std::string extension = path.substr(dot + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value)
                   {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension;
}

Audio LoadMp3Mono(const std::string& path)
{
    drmp3_config config;
    drmp3_uint64 total_frames = 0;
    float* pcm = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &config, &total_frames, nullptr);
    if (pcm == nullptr)
    {
        throw std::runtime_error("failed to decode mp3 file: " + path);
    }

    Audio audio;
    audio.sample_rate = static_cast<int>(config.sampleRate);
    audio.samples.resize(total_frames);
    const auto channels = config.channels == 0 ? 1U : config.channels;
    for (drmp3_uint64 i = 0; i < total_frames; ++i)
    {
        float sum = 0.0F;
        for (unsigned channel = 0; channel < channels; ++channel)
        {
            sum += pcm[i * channels + channel];
        }
        audio.samples[i] = sum / static_cast<float>(channels);
    }
    drmp3_free(pcm, nullptr);

    if (audio.samples.empty() || audio.sample_rate == 0)
    {
        throw std::runtime_error("empty mp3 file: " + path);
    }
    return audio;
}

Audio Resample(const Audio& input, int target_rate)
{
    if (input.sample_rate == target_rate || input.samples.empty())
    {
        return input;
    }

    const auto source_rate = static_cast<double>(input.sample_rate);
    const auto destination_count = static_cast<size_t>(input.samples.size() * target_rate / source_rate);
    Audio output;
    output.sample_rate = target_rate;
    output.samples.resize(destination_count);

    const double step = source_rate / target_rate;
    const auto last = input.samples.size() - 1;
    for (size_t i = 0; i < destination_count; ++i)
    {
        if (step > 1.0)
        {
            const auto begin = static_cast<size_t>(i * step);
            const auto end = std::min(static_cast<size_t>((i + 1) * step), input.samples.size());
            float sum = 0.0F;
            for (size_t j = begin; j < end; ++j)
            {
                sum += input.samples[j];
            }
            output.samples[i] = end > begin ? sum / static_cast<float>(end - begin) : input.samples[begin];
        }
        else
        {
            const double position = i * step;
            const auto index = std::min(static_cast<size_t>(position), last);
            const auto next = std::min(index + 1, last);
            const auto fraction = static_cast<float>(position - index);
            output.samples[i] = input.samples[index] * (1.0F - fraction) + input.samples[next] * fraction;
        }
    }
    return output;
}

}  // namespace

Audio LoadAudio(const std::string& path, int target_rate)
{
    const auto extension = LowercaseExtension(path);
    Audio audio = extension == "mp3" ? LoadMp3Mono(path) : LoadWavMono(path);
    return Resample(audio, target_rate);
}

Audio LoadWavMono(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open audio file: " + path);
    }
    if (ReadTag(stream) != "RIFF")
    {
        throw std::runtime_error("expected RIFF wav file");
    }
    ReadU32(stream);
    if (ReadTag(stream) != "WAVE")
    {
        throw std::runtime_error("expected WAVE file");
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> data;

    while (stream && (!format || data.empty()))
    {
        const std::string chunk = ReadTag(stream);
        const uint32_t size = ReadU32(stream);
        if (chunk == "fmt ")
        {
            format = ReadU16(stream);
            channels = ReadU16(stream);
            sample_rate = ReadU32(stream);
            ReadU32(stream);
            ReadU16(stream);
            bits_per_sample = ReadU16(stream);
            if (size > 16)
            {
                stream.seekg(size - 16, std::ios::cur);
            }
        }
        else if (chunk == "data")
        {
            data.resize(size);
            stream.read(data.data(), size);
        }
        else
        {
            stream.seekg(size, std::ios::cur);
        }
        if (size % 2 == 1)
        {
            stream.seekg(1, std::ios::cur);
        }
    }

    if (channels == 0 || sample_rate == 0 || data.empty())
    {
        throw std::runtime_error("incomplete wav file");
    }

    const int bytes = bits_per_sample / 8;
    const size_t frames = data.size() / (bytes * channels);
    Audio audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    audio.samples.resize(frames);

    for (size_t i = 0; i < frames; ++i)
    {
        double sum = 0.0;
        for (uint16_t channel = 0; channel < channels; ++channel)
        {
            const char* pointer = data.data() + (i * channels + channel) * bytes;
            if (format == 1 && bits_per_sample == 16)
            {
                int16_t value = 0;
                std::copy(pointer, pointer + 2, reinterpret_cast<char*>(&value));
                sum += static_cast<double>(value) / 32768.0;
            }
            else if (format == 3 && bits_per_sample == 32)
            {
                float value = 0.0F;
                std::copy(pointer, pointer + 4, reinterpret_cast<char*>(&value));
                sum += value;
            }
            else
            {
                throw std::runtime_error("only PCM16 and float32 wav are supported");
            }
        }
        audio.samples[i] = static_cast<float>(sum / channels);
    }
    return audio;
}

}  // namespace din::io
