// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit test for din::io::Tokenizer::Encode on three tiny tokenizer.json layouts built in code:
//   * Gemma style (SentencePiece-style BPE): Replace(" " -> "▁"), byte_fallback, <bos> template,
//     special and non-special added tokens, no dummy prefix.
//   * Llama style: the same with a Prepend("▁") normalizer.
//   * GPT-2 / Qwen style byte-level BPE (regression for the existing path).
// The expected ids were produced by Hugging Face `tokenizers` 0.23.2 from identical tokenizer.json content.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tokenizer.h"
#include <nlohmann/json.hpp>

namespace
{

using nlohmann::json;

constexpr const char* kSpace = "\xE2\x96\x81";  // U+2581, SentencePiece space marker

struct Case
{
    std::string text;
    bool add_special_tokens;
    std::vector<int64_t> ids;
};

std::string utf8(uint32_t code)
{
    std::string out;
    if (code < 0x80)
    {
        out.push_back(static_cast<char>(code));
    }
    else if (code < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    return out;
}

json added_token(int64_t id, const std::string& content, bool special)
{
    return {{"id", id},          {"content", content}, {"single_word", false}, {"lstrip", false},
            {"rstrip", false},   {"normalized", false}, {"special", special}};
}

// Vocab ids: <pad> <eos> <bos> <unk> \n \n\n <i> a b c d e . ▁ ab bc abc ▁a ▁ab aa, then <0x00>..<0xFF> from 20.
json sentencepiece_tokenizer(bool prepend)
{
    const std::string sp = kSpace;
    std::vector<std::string> tokens = {"<pad>", "<eos>", "<bos>", "<unk>", "\n", "\n\n", "<i>", "a", "b", "c", "d",
                                       "e",     ".",     sp,      "ab",    "bc", "abc",  sp + "a", sp + "ab", "aa"};
    for (int b = 0; b < 256; ++b)
    {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        tokens.emplace_back(name);
    }
    json vocab = json::object();
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        vocab[tokens[i]] = static_cast<int64_t>(i);
    }
    // Rank order matters: "a b" before "b c"; "a a" last so ties resolve leftmost.
    const json merges = json::array({{"a", "b"}, {"b", "c"}, {"ab", "c"}, {sp, "a"}, {sp + "a", "b"}, {"a", "a"}});

    const json replace = {{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", sp}};
    const json normalizer =
        prepend ? json{{"type", "Sequence"}, {"normalizers", {{{"type", "Prepend"}, {"prepend", sp}}, replace}}} : replace;

    return {
        {"version", "1.0"},
        {"added_tokens",
         {added_token(0, "<pad>", true), added_token(1, "<eos>", true), added_token(2, "<bos>", true),
          added_token(3, "<unk>", true), added_token(4, "\n", false), added_token(5, "\n\n", false),
          added_token(6, "<i>", false)}},
        {"normalizer", normalizer},
        {"pre_tokenizer",
         prepend ? json(nullptr)
                 : json{{"type", "Split"}, {"pattern", {{"String", " "}}}, {"behavior", "MergedWithPrevious"},
                        {"invert", false}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single", {{{"SpecialToken", {{"id", "<bos>"}, {"type_id", 0}}}}, {{"Sequence", {{"id", "A"}, {"type_id", 0}}}}}},
          {"special_tokens", {{"<bos>", {{"id", "<bos>"}, {"ids", {2}}, {"tokens", {"<bos>"}}}}}}}},
        {"model",
         {{"type", "BPE"}, {"unk_token", "<unk>"}, {"fuse_unk", true}, {"byte_fallback", true}, {"ignore_merges", false},
          {"vocab", vocab}, {"merges", merges}}},
    };
}

// GPT-2 byte-level alphabet (256 code points) in code-point order, then merged pieces.
json byte_level_tokenizer()
{
    std::vector<uint32_t> alphabet;
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b)
    {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        alphabet.push_back(printable ? b : 256 + extra++);
    }
    std::sort(alphabet.begin(), alphabet.end());
    const std::string g = utf8(0x120);  // byte-level space
    const std::vector<std::pair<std::string, std::string>> merge_list = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {g, "w"}, {g + "w", "o"},
        {"a", "b"}, {"ab", "c"}, {g, "a"},    {g + "a", "b"}, {g + "ab", "c"}};
    json vocab = json::object();
    int64_t id = 0;
    for (const uint32_t code : alphabet)
    {
        vocab[utf8(code)] = id++;
    }
    json merges = json::array();
    for (const auto& [left, right] : merge_list)
    {
        merges.push_back({left, right});
        if (!vocab.contains(left + right))
        {
            vocab[left + right] = id++;
        }
    }
    return {
        {"version", "1.0"},
        {"added_tokens", json::array()},
        {"normalizer", nullptr},
        {"pre_tokenizer", {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", true}}},
        {"post_processor", nullptr},
        {"model", {{"type", "BPE"}, {"byte_fallback", false}, {"vocab", vocab}, {"merges", merges}}},
    };
}

// clang-format off
const std::vector<Case> kGemmaStyleCases = {
    {"", true, {2}},
    {"", false, {}},
    {"abc", true, {2, 16}},
    {"abc", false, {16}},
    {"aaa", true, {2, 19, 7}},
    {"aaa", false, {19, 7}},
    {"a b", true, {2, 7, 13, 8}},
    {"a b", false, {7, 13, 8}},
    {"  ab", true, {2, 13, 13, 14}},
    {"  ab", false, {13, 13, 14}},
    {"ab ", true, {2, 14, 13}},
    {"ab ", false, {14, 13}},
    {"a  b", true, {2, 7, 13, 13, 8}},
    {"a  b", false, {7, 13, 13, 8}},
    {"a\xC3\xA9", true, {2, 7, 215, 189}},
    {"a\xC3\xA9", false, {7, 215, 189}},
    {"a\xF0\x9F\xA6\x8A", true, {2, 7, 260, 179, 186, 158}},
    {"a\xF0\x9F\xA6\x8A", false, {7, 260, 179, 186, 158}},
    {"x", true, {2, 140}},
    {"x", false, {140}},
    {"a\nb", true, {2, 7, 4, 8}},
    {"a\nb", false, {7, 4, 8}},
    {"a\n\nb", true, {2, 7, 5, 8}},
    {"a\n\nb", false, {7, 5, 8}},
    {"a\n\n\nb", true, {2, 7, 5, 4, 8}},
    {"a\n\n\nb", false, {7, 5, 4, 8}},
    {"<i>ab</i>", true, {2, 6, 14, 80, 67, 125, 82}},
    {"<i>ab</i>", false, {6, 14, 80, 67, 125, 82}},
    {"a<eos>b", true, {2, 7, 1, 8}},
    {"a<eos>b", false, {7, 1, 8}},
    {"ab c.\n\n<bos>a\xC3\xA9", true, {2, 14, 13, 9, 12, 5, 2, 7, 215, 189}},
    {"ab c.\n\n<bos>a\xC3\xA9", false, {14, 13, 9, 12, 5, 2, 7, 215, 189}},
};

const std::vector<Case> kLlamaStyleCases = {
    {"", true, {2}},
    {"", false, {}},
    {"abc", true, {2, 13, 16}},
    {"abc", false, {13, 16}},
    {"aaa", true, {2, 17, 19}},
    {"aaa", false, {17, 19}},
    {"a b", true, {2, 17, 13, 8}},
    {"a b", false, {17, 13, 8}},
    {"  ab", true, {2, 13, 13, 13, 14}},
    {"  ab", false, {13, 13, 13, 14}},
    {"ab ", true, {2, 13, 14, 13}},
    {"ab ", false, {13, 14, 13}},
    {"a  b", true, {2, 17, 13, 13, 8}},
    {"a  b", false, {17, 13, 13, 8}},
    {"a\xC3\xA9", true, {2, 17, 215, 189}},
    {"a\xC3\xA9", false, {17, 215, 189}},
    {"a\xF0\x9F\xA6\x8A", true, {2, 17, 260, 179, 186, 158}},
    {"a\xF0\x9F\xA6\x8A", false, {17, 260, 179, 186, 158}},
    {"x", true, {2, 13, 140}},
    {"x", false, {13, 140}},
    {"a\nb", true, {2, 17, 4, 13, 8}},
    {"a\nb", false, {17, 4, 13, 8}},
    {"a\n\nb", true, {2, 17, 5, 13, 8}},
    {"a\n\nb", false, {17, 5, 13, 8}},
    {"a\n\n\nb", true, {2, 17, 5, 4, 13, 8}},
    {"a\n\n\nb", false, {17, 5, 4, 13, 8}},
    {"<i>ab</i>", true, {2, 6, 13, 14, 80, 67, 125, 82}},
    {"<i>ab</i>", false, {6, 13, 14, 80, 67, 125, 82}},
    {"a<eos>b", true, {2, 17, 1, 13, 8}},
    {"a<eos>b", false, {17, 1, 13, 8}},
    {"ab c.\n\n<bos>a\xC3\xA9", true, {2, 13, 14, 13, 9, 12, 5, 2, 17, 215, 189}},
    {"ab c.\n\n<bos>a\xC3\xA9", false, {13, 14, 13, 9, 12, 5, 2, 17, 215, 189}},
};

// The byte-level path ignores add_special_tokens; this tokenizer has no special tokens.
const std::vector<Case> kByteLevelCases = {
    {"", false, {}},
    {"abc abc", false, {263, 220, 263}},
    {"hello world", false, {259, 261, 81, 75, 67}},
    {"a,b. c", false, {64, 11, 65, 13, 220, 66}},
};
// clang-format on

std::string escape(const std::string& text)
{
    std::ostringstream out;
    for (const unsigned char c : text)
    {
        if (c == '\n')
            out << "\\n";
        else if (c < 0x20 || c >= 0x7F)
        {
            char hex[8];
            std::snprintf(hex, sizeof(hex), "\\x%02X", c);
            out << hex;
        }
        else
            out << c;
    }
    return out.str();
}

std::string join(const std::vector<int64_t>& ids)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        out << (i ? ", " : "") << ids[i];
    }
    out << "]";
    return out.str();
}

// din::io::Tokenizer loads from a path, so each tokenizer.json is written to a temporary directory.
int run_cases(const std::filesystem::path& dir, const char* name, const json& tokenizer_json,
              const std::vector<Case>& cases)
{
    const auto path = dir / name;
    std::ofstream(path, std::ios::binary) << tokenizer_json.dump();
    const din::io::Tokenizer tokenizer(path.string(), din::io::TokenizerFormat::Json);
    int failures = 0;
    for (const auto& test_case : cases)
    {
        const auto got = tokenizer.Encode(test_case.text, test_case.add_special_tokens);
        if (got != test_case.ids)
        {
            ++failures;
            std::cerr << "FAIL " << name << " \"" << escape(test_case.text)
                      << "\" add_special_tokens=" << test_case.add_special_tokens << "\n  expected "
                      << join(test_case.ids) << "\n  got      " << join(got) << std::endl;
        }
    }
    std::cout << name << ": " << (cases.size() - failures) << "/" << cases.size() << " cases match" << std::endl;
    return failures;
}

}  // namespace

int main()
{
    try
    {
        const auto dir = std::filesystem::temp_directory_path() / "din_tokenizer_test";
        std::filesystem::create_directories(dir);
        int failures = 0;
        failures += run_cases(dir, "gemma_style.json", sentencepiece_tokenizer(false), kGemmaStyleCases);
        failures += run_cases(dir, "llama_style.json", sentencepiece_tokenizer(true), kLlamaStyleCases);
        failures += run_cases(dir, "byte_level.json", byte_level_tokenizer(), kByteLevelCases);
        std::filesystem::remove_all(dir);
        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
