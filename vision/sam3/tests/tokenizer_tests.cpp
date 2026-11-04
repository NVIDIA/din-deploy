// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_tokenizer.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    if (!std::filesystem::exists(argv[1]) || !std::filesystem::exists(argv[2])) {
        std::cout << "SKIP: export tokenizer fixtures first\n"; return 77;
    }
    try {
        din::sam3::TextTokenizer tokenizer(argv[1]);
        std::ifstream input(argv[2]);
        auto fixtures = nlohmann::json::parse(input);
        int failed = 0;
        for (const auto& fixture : fixtures) {
            auto actual = tokenizer.Encode(fixture.at("text"));
            auto expected = fixture.at("ids").get<std::vector<int64_t>>();
            if (actual != expected) {
                std::cerr << "FAIL " << fixture.at("text") << "\nactual " << nlohmann::json(actual)
                          << "\nexpected " << nlohmann::json(expected) << '\n'; ++failed;
            }
        }
        std::cout << fixtures.size() - failed << '/' << fixtures.size() << " tokenizer cases passed\n";
        return failed ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
