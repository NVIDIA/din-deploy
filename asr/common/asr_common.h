// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace din::asr::common
{

// --- file helpers --------------------------------------------------------

inline std::string ReadText(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

inline std::string RequirePath(const std::filesystem::path& path, std::string_view name)
{
    if (!std::filesystem::exists(path))
    {
        throw std::runtime_error("missing " + std::string{name} + ": " + path.string());
    }
    return path.string();
}

inline std::vector<float> ReadFloatFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        return {};
    }
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
    stream.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}

// --- timing helper -------------------------------

inline double SecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// --- Shared command-line front end --------------------------------------

struct TranscriptionOptions
{
    std::string timestamps = "none";
};

}  // namespace din::asr::common
