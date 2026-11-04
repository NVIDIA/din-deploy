// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace din::sam3 {
class TextTokenizer {
public:
    explicit TextTokenizer(const std::filesystem::path& path);
    std::vector<int64_t> Encode(const std::string& text) const;
private:
    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<std::string, size_t> ranks_;
    std::unordered_map<std::string, std::string> replacements_, entities_;
    std::array<std::string, 256> bytes_;
    size_t context_ = 0;
    int64_t start_ = 0, end_ = 0;
};
}
