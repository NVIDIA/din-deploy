// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_tokenizer.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <icu.h>
#else
#include <unicode/ustring.h>
#include <unicode/unorm2.h>
#include <unicode/uregex.h>
#endif

namespace din::sam3 {
namespace {
void Check(UErrorCode status) {
    if (U_FAILURE(status)) throw std::runtime_error("SAM3 Unicode processing failed: " + std::to_string(status));
}
std::u16string Utf16(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t size = 0;
    u_strFromUTF8(nullptr, 0, &size, text.data(), static_cast<int32_t>(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) Check(status);
    status = U_ZERO_ERROR;
    std::u16string result(size, u'\0');
    u_strFromUTF8(reinterpret_cast<UChar*>(result.data()), size, nullptr, text.data(), static_cast<int32_t>(text.size()), &status);
    Check(status); return result;
}
std::string Utf8(const std::u16string& text) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t size = 0;
    u_strToUTF8(nullptr, 0, &size, reinterpret_cast<const UChar*>(text.data()), static_cast<int32_t>(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) Check(status);
    status = U_ZERO_ERROR;
    std::string result(size, '\0');
    u_strToUTF8(result.data(), size, nullptr, reinterpret_cast<const UChar*>(text.data()), static_cast<int32_t>(text.size()), &status);
    Check(status); return result;
}
std::u16string NormalizeLower(const std::string& text) {
    auto input = Utf16(text);
    UErrorCode status = U_ZERO_ERROR;
    const auto* normalizer = unorm2_getNFCInstance(&status);
    Check(status);
    int32_t size = unorm2_normalize(normalizer, reinterpret_cast<const UChar*>(input.data()), static_cast<int32_t>(input.size()), nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) Check(status);
    std::u16string normalized(size, u'\0');
    status = U_ZERO_ERROR;
    unorm2_normalize(normalizer, reinterpret_cast<const UChar*>(input.data()), static_cast<int32_t>(input.size()), reinterpret_cast<UChar*>(normalized.data()), size, &status);
    Check(status); status = U_ZERO_ERROR;
    size = u_strToLower(nullptr, 0, reinterpret_cast<const UChar*>(normalized.data()), static_cast<int32_t>(normalized.size()), "", &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) Check(status);
    std::u16string lowered(size, u'\0');
    status = U_ZERO_ERROR;
    u_strToLower(reinterpret_cast<UChar*>(lowered.data()), size, reinterpret_cast<const UChar*>(normalized.data()), static_cast<int32_t>(normalized.size()), "", &status);
    Check(status); return lowered;
}
std::string Unescape(std::string input, const std::unordered_map<std::string, std::string>& entities) {
    std::string result;
    for (size_t i = 0; i < input.size();) {
        if (input[i] != '&') {result += input[i++]; continue;}
        if (i + 2 < input.size() && input[i + 1] == '#') {
            size_t j = i + 2;
            int base = 10;
            if (input[j] == 'x' || input[j] == 'X') {base = 16; ++j;}
            const size_t start = j;
            uint32_t code = 0;
            while (j < input.size()) {
                const unsigned char c = input[j];
                const int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16;
                if (digit >= base) break;
                code = std::min(0x110000u, code * base + digit); ++j;
            }
            if (j != start) {
                if (j < input.size() && input[j] == ';') ++j;
                auto special = entities.find("#" + std::to_string(code) + ";");
                if (special != entities.end()) result += special->second;
                else if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) result += "\xef\xbf\xbd";
                else if ((code & 0xffff) >= 0xfffe || (code >= 0xfdd0 && code <= 0xfdef)) { /* HTML discards noncharacters. */ }
                else if (code <= 0xffff) result += Utf8(std::u16string(1, static_cast<char16_t>(code)));
                else {
                    code -= 0x10000;
                    result += Utf8(std::u16string{static_cast<char16_t>(0xd800 + (code >> 10)), static_cast<char16_t>(0xdc00 + (code & 1023))});
                }
                i = j; continue;
            }
        }
        size_t length = 0;
        const std::string* replacement = nullptr;
        for (size_t n = 1; n <= 32 && i + n < input.size(); ++n) {
            const auto found = entities.find(input.substr(i + 1, n));
            if (found != entities.end()) {length = n + 1; replacement = &found->second;}
        }
        if (replacement) {result += *replacement; i += length;}
        else {result += input[i++];}
    }
    return result;
}
}

TextTokenizer::TextTokenizer(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open SAM3 tokenizer: " + path.string());
    const auto data = nlohmann::json::parse(stream);
    vocab_ = data.at("vocab").get<decltype(vocab_)>();
    replacements_ = data.value("normalization_map", decltype(replacements_){});
    entities_ = data.value("html_entities", decltype(entities_){});
    context_ = data.at("context_length"); start_ = data.at("start_token"); end_ = data.at("end_token");
    if (context_ < 2 || context_ > 4096) throw std::runtime_error("Invalid SAM3 text context length");
    for (const auto& pair : data.at("merges")) {
        auto key = pair.at(0).get<std::string>() + '\0' + pair.at(1).get<std::string>();
        ranks_.emplace(std::move(key), ranks_.size());
    }
    int extra = 0;
    for (int i = 0; i < 256; ++i) {
        const bool literal = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
        bytes_[i] = Utf8(std::u16string(1, static_cast<char16_t>(literal ? i : 256 + extra++)));
    }
}

std::vector<int64_t> TextTokenizer::Encode(const std::string& text) const {
    // ftfy repeatedly decodes HTML entities unless the input looks like markup.
    std::string html_cleaned = text;
    if (text.find('<') == std::string::npos) {
        for (;;) {
            auto decoded = Unescape(html_cleaned, entities_);
            if (decoded == html_cleaned) break;
            html_cleaned = std::move(decoded);
        }
    }
    std::string cleaned;
    for (size_t i = 0; i < html_cleaned.size();) {
        const auto lead = static_cast<unsigned char>(html_cleaned[i]);
        const size_t width = lead < 128 ? 1 : lead < 224 ? 2 : lead < 240 ? 3 : 4;
        auto character = html_cleaned.substr(i, width);
        const auto replacement = replacements_.find(character);
        cleaned += replacement == replacements_.end() ? character : replacement->second;
        i += width;
    }
    cleaned = Unescape(Unescape(cleaned, entities_), entities_);
    const auto input = NormalizeLower(cleaned);
    const std::u16string pattern = u"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<URegularExpression, decltype(&uregex_close)> regex(
        uregex_open(reinterpret_cast<const UChar*>(pattern.data()), static_cast<int32_t>(pattern.size()), 0, nullptr, &status), uregex_close);
    Check(status);
    uregex_setText(regex.get(), reinterpret_cast<const UChar*>(input.data()), static_cast<int32_t>(input.size()), &status);
    Check(status);
    std::vector<int64_t> ids{start_};
    while (uregex_findNext(regex.get(), &status)) {
        const auto begin = uregex_start(regex.get(), 0, &status), end = uregex_end(regex.get(), 0, &status);
        Check(status);
        const auto piece = Utf8(input.substr(begin, end - begin));
        if (piece == "<start_of_text>" || piece == "<end_of_text>") {ids.push_back(vocab_.at(piece)); continue;}
        std::vector<std::string> symbols;
        for (unsigned char byte : piece) symbols.push_back(bytes_[byte]);
        if (symbols.empty()) continue;
        symbols.back() += "</w>";
        while (symbols.size() > 1) {
            size_t best = std::numeric_limits<size_t>::max();
            std::string left, right;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto found = ranks_.find(symbols[i] + '\0' + symbols[i + 1]);
                if (found != ranks_.end() && found->second < best) {best = found->second; left = symbols[i]; right = symbols[i + 1];}
            }
            if (best == std::numeric_limits<size_t>::max()) break;
            std::vector<std::string> merged;
            for (size_t i = 0; i < symbols.size(); ++i) {
                if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {merged.push_back(left + right); ++i;}
                else merged.push_back(symbols[i]);
            }
            symbols = std::move(merged);
        }
        for (const auto& symbol : symbols) ids.push_back(vocab_.at(symbol));
    }
    Check(status);
    ids.push_back(end_);
    if (ids.size() > context_) {ids.resize(context_); ids.back() = end_;}
    ids.resize(context_, 0);
    return ids;
}
}
