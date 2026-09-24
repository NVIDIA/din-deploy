// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../text_output.h"
#include <nlohmann/json.hpp>

int main()
{
    using din::asr::qwen3::detail::TrimIncompleteUtf8Suffix;
    try
    {
        // Cover every truncation of two-, three-, and four-byte characters.
        const std::vector<std::string> characters = {"\xC2\xA2", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80"};
        for (const auto& character : characters)
        {
            for (size_t length = 0; length <= character.size(); ++length)
            {
                const auto prefix = std::string("earlier chunk ") + characters[1] + " ";
                auto text = prefix + character.substr(0, length);
                TrimIncompleteUtf8Suffix(text);
                const auto expected = prefix + (length == character.size() ? character : "");
                if (text != expected)
                    throw std::runtime_error("Trimming lost complete transcript text");
                const nlohmann::json output{{"transcription", text}, {"reached_eos", false}};
                const auto parsed = nlohmann::json::parse(output.dump());
                if (parsed.at("transcription") != expected || parsed.at("reached_eos") != false)
                    throw std::runtime_error("Partial transcript did not survive JSON serialization");
            }
        }
        // Do not hide other encoding errors by removing arbitrary trailing bytes.
        for (const std::string original :
             {"", "ASCII", "\x80", "\xFFtext", "\xC0", "\xE0\x80", "\xED\xA0", "\xF0\x80", "\xF4\x90", "\xF5\x80"})
        {
            auto text = original;
            TrimIncompleteUtf8Suffix(text);
            if (text != original)
                throw std::runtime_error("Modified complete text or an unrelated encoding error");
        }
        std::cout << "Qwen3 text output checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
