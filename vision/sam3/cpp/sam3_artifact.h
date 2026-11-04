// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>

namespace din::sam3 {
// Keep EP contexts separate when a graph or its weight sidecar is re-exported.
// The common runtime also validates contexts, but a changed input shape must not
// select a previous graph merely because its ONNX filename stayed the same.
inline std::filesystem::path ArtifactCache(const std::filesystem::path& root, const std::filesystem::path& model) {
    std::string identity = std::filesystem::absolute(model).lexically_normal().generic_string();
    for (const auto& path : {model, std::filesystem::path(model.string() + ".data")}) {
        if (std::filesystem::exists(path)) {
            identity += ":" + std::to_string(std::filesystem::file_size(path));
            identity += ":" + std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
        }
    }
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : identity) {hash ^= c; hash *= 1099511628211ull;}
    std::ostringstream name; name << std::hex << hash;
    return root / name.str();
}
}
