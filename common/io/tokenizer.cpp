// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace din::io
{
namespace
{

bool StartsWith(std::string_view value, std::string_view prefix)
{
    return value.substr(0, prefix.size()) == prefix;
}

std::string BpePairKey(std::string_view left, std::string_view right)
{
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\x1F');
    key.append(right);
    return key;
}

std::string Utf8Encode(uint32_t code)
{
    std::string text;
    if (code <= 0x7F)
    {
        text.push_back(static_cast<char>(code));
    }
    else if (code <= 0x7FF)
    {
        text.push_back(static_cast<char>(0xC0 | (code >> 6)));
        text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    else if (code <= 0xFFFF)
    {
        text.push_back(static_cast<char>(0xE0 | (code >> 12)));
        text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    else
    {
        text.push_back(static_cast<char>(0xF0 | (code >> 18)));
        text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    return text;
}

// GPT-2 byte-level BPE maps each raw byte to a printable Unicode code point.
std::array<std::string, 256> BuildByteEncoder()
{
    std::vector<uint32_t> printable;
    for (uint32_t c = U'!'; c <= U'~'; ++c)
    {
        printable.push_back(c);
    }
    for (uint32_t c = 0xA1; c <= 0xAC; ++c)
    {
        printable.push_back(c);
    }
    for (uint32_t c = 0xAE; c <= 0xFF; ++c)
    {
        printable.push_back(c);
    }

    std::array<std::string, 256> byte_to_code;
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b)
    {
        uint32_t code = b;
        if (std::find(printable.begin(), printable.end(), b) == printable.end())
        {
            code = 256 + extra;
            ++extra;
        }
        byte_to_code[b] = Utf8Encode(code);
    }
    return byte_to_code;
}

struct JsonTokenizerData
{
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int64_t> token_to_id;
    std::unordered_map<std::string, int32_t> bpe_ranks;
    std::vector<std::string> special_tokens;
};

void AddToken(JsonTokenizerData& data, const std::string& token, int64_t id)
{
    if (id < 0)
    {
        return;
    }
    if (static_cast<size_t>(id) >= data.id_to_token.size())
    {
        data.id_to_token.resize(static_cast<size_t>(id + 1));
    }
    data.id_to_token[static_cast<size_t>(id)] = token;
    data.token_to_id[token] = id;
}

JsonTokenizerData LoadJsonTokenizerData(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open tokenizer file: " + path);
    }
    const auto data = nlohmann::json::parse(stream);
    if (!data.contains("model") || !data["model"].contains("vocab"))
    {
        throw std::runtime_error("tokenizer.json is missing model.vocab");
    }

    JsonTokenizerData tokenizer_data;

    const auto& vocab = data["model"]["vocab"];
    for (const auto& [token, id_val] : vocab.items())
    {
        AddToken(tokenizer_data, token, id_val.get<int64_t>());
    }

    if (data.contains("added_tokens"))
    {
        for (const auto& token : data["added_tokens"])
        {
            if (!token.contains("id") || !token.contains("content"))
            {
                continue;
            }
            const auto content = token["content"].get<std::string>();
            AddToken(tokenizer_data, content, token["id"].get<int64_t>());
            if (token.value("special", false))
            {
                tokenizer_data.special_tokens.push_back(content);
            }
        }
    }

    if (data["model"].contains("merges"))
    {
        int32_t rank = 0;
        for (const auto& merge : data["model"]["merges"])
        {
            std::string left;
            std::string right;
            if (merge.is_string())
            {
                const auto text = merge.get<std::string>();
                const auto split = text.find(' ');
                if (split == std::string::npos)
                {
                    continue;
                }
                left = text.substr(0, split);
                right = text.substr(split + 1);
            }
            else if (merge.is_array() && merge.size() == 2)
            {
                left = merge[0].get<std::string>();
                right = merge[1].get<std::string>();
            }
            else
            {
                continue;
            }
            tokenizer_data.bpe_ranks.emplace(BpePairKey(left, right), rank);
            ++rank;
        }
    }

    std::sort(tokenizer_data.special_tokens.begin(), tokenizer_data.special_tokens.end(),
              [](const auto& left, const auto& right)
              {
                  return left.size() > right.size();
              });
    return tokenizer_data;
}

std::vector<std::string> LoadFlatVocab(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open vocab file: " + path);
    }
    std::vector<std::string> id_to_token;
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        id_to_token.push_back(line);
    }
    return id_to_token;
}

// Whisper stores vocab entries in byte-character form; decoding reverses it.
std::unordered_map<uint32_t, uint8_t> BuildByteDecoder()
{
    std::vector<uint32_t> printable;
    for (uint32_t c = U'!'; c <= U'~'; ++c)
    {
        printable.push_back(c);
    }
    for (uint32_t c = 0xA1; c <= 0xAC; ++c)
    {
        printable.push_back(c);
    }
    for (uint32_t c = 0xAE; c <= 0xFF; ++c)
    {
        printable.push_back(c);
    }

    std::unordered_map<uint32_t, uint8_t> code_to_byte;
    for (uint32_t b = 0; b < 256; ++b)
    {
        if (std::find(printable.begin(), printable.end(), b) != printable.end())
        {
            code_to_byte.emplace(b, static_cast<uint8_t>(b));
        }
    }
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b)
    {
        if (std::find(printable.begin(), printable.end(), b) == printable.end())
        {
            code_to_byte.emplace(256 + extra, static_cast<uint8_t>(b));
            ++extra;
        }
    }
    return code_to_byte;
}

uint32_t NextCodePoint(const std::string& text, size_t& i)
{
    const auto byte = static_cast<uint8_t>(text[i]);
    uint32_t code = 0;
    size_t extra = 0;
    if (byte < 0x80)
    {
        code = byte;
    }
    else if ((byte & 0xE0) == 0xC0)
    {
        code = byte & 0x1F;
        extra = 1;
    }
    else if ((byte & 0xF0) == 0xE0)
    {
        code = byte & 0x0F;
        extra = 2;
    }
    else
    {
        code = byte & 0x07;
        extra = 3;
    }
    ++i;
    for (size_t k = 0; k < extra && i < text.size(); ++k, ++i)
    {
        code = (code << 6) | (static_cast<uint8_t>(text[i]) & 0x3F);
    }
    return code;
}

std::vector<std::string> LoadWhisperJsonVocab(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open vocab.json: " + path);
    }
    const auto vocab = nlohmann::json::parse(stream);
    const auto code_to_byte = BuildByteDecoder();

    int64_t max_id = -1;
    for (const auto& [token, id_value] : vocab.items())
    {
        max_id = std::max(max_id, id_value.get<int64_t>());
    }

    std::vector<std::string> id_to_token(static_cast<size_t>(max_id + 1));
    for (const auto& [token, id_value] : vocab.items())
    {
        const int64_t id = id_value.get<int64_t>();
        if (id >= kWhisperTextTokenLimit)
        {
            continue;
        }

        std::string bytes;
        bytes.reserve(token.size());
        for (size_t i = 0; i < token.size();)
        {
            const uint32_t code = NextCodePoint(token, i);
            const auto it = code_to_byte.find(code);
            if (it != code_to_byte.end())
            {
                bytes.push_back(static_cast<char>(it->second));
            }
        }
        id_to_token[static_cast<size_t>(id)] = std::move(bytes);
    }
    return id_to_token;
}

std::string StripLangTags(const std::string& text)
{
    return std::regex_replace(text, std::regex("<[^>\\s]+>"), "");
}

constexpr std::array<const char*, 99> kWhisperLanguageCodes = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt",  "tr", "pl", "ca", "nl", "ar", "sv", "it", "id",
    "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",  "da", "hu", "ta", "no", "th", "ur", "hr", "bg",
    "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv",  "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",  "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
    "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",  "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt",
    "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"};

}  // namespace

Tokenizer::Tokenizer(const std::string& path, TokenizerFormat format)
{
    switch (format)
    {
    case TokenizerFormat::Json:
    {
        auto tokenizer_data = LoadJsonTokenizerData(path);
        id_to_token_ = std::move(tokenizer_data.id_to_token);
        token_to_id_ = std::move(tokenizer_data.token_to_id);
        bpe_ranks_ = std::move(tokenizer_data.bpe_ranks);
        special_tokens_ = std::move(tokenizer_data.special_tokens);
        byte_encoder_ = BuildByteEncoder();
        decode_mode_ = DecodeMode::Pieces;
        break;
    }
    case TokenizerFormat::Vocab:
        id_to_token_ = LoadFlatVocab(path);
        for (size_t i = 0; i < id_to_token_.size(); ++i)
        {
            token_to_id_[id_to_token_[i]] = static_cast<int64_t>(i);
        }
        decode_mode_ = DecodeMode::Pieces;
        break;
    case TokenizerFormat::WhisperJson:
        id_to_token_ = LoadWhisperJsonVocab(path);
        for (size_t i = 0; i < id_to_token_.size(); ++i)
        {
            token_to_id_[id_to_token_[i]] = static_cast<int64_t>(i);
        }
        lang_codes_.assign(kWhisperLanguageCodes.begin(), kWhisperLanguageCodes.end());
        decode_mode_ = DecodeMode::WhisperByteBpe;
        break;
    }
}

const std::string& Tokenizer::Token(int64_t id) const
{
    static const std::string empty;
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size())
    {
        return empty;
    }
    return id_to_token_[static_cast<size_t>(id)];
}

namespace
{

bool IsAsciiLetter(uint8_t c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool IsAsciiDigit(uint8_t c)
{
    return c >= '0' && c <= '9';
}

bool IsAsciiWhitespace(uint8_t c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool IsContractionAt(std::string_view text, size_t pos, size_t* length)
{
    constexpr std::array<std::string_view, 7> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (const auto contraction : contractions)
    {
        if (pos + contraction.size() > text.size())
        {
            continue;
        }
        bool match = true;
        for (size_t i = 0; i < contraction.size(); ++i)
        {
            const auto left = static_cast<unsigned char>(text[pos + i]);
            const auto right = static_cast<unsigned char>(contraction[i]);
            if (std::tolower(left) != std::tolower(right))
            {
                match = false;
                break;
            }
        }
        if (match)
        {
            *length = contraction.size();
            return true;
        }
    }
    return false;
}

std::vector<std::string> SplitForByteLevelBpe(std::string_view text)
{
    std::vector<std::string> pieces;
    size_t pos = 0;
    while (pos < text.size())
    {
        size_t contraction_len = 0;
        if (IsContractionAt(text, pos, &contraction_len))
        {
            pieces.emplace_back(text.substr(pos, contraction_len));
            pos += contraction_len;
            continue;
        }

        const auto current = static_cast<uint8_t>(text[pos]);
        if (current == ' ' && pos + 1 < text.size() &&
            (IsAsciiLetter(static_cast<uint8_t>(text[pos + 1])) || IsAsciiDigit(static_cast<uint8_t>(text[pos + 1])) ||
             !IsAsciiWhitespace(static_cast<uint8_t>(text[pos + 1]))))
        {
            const size_t start = pos++;
            const auto next = static_cast<uint8_t>(text[pos]);
            if (IsAsciiLetter(next))
            {
                while (pos < text.size() && IsAsciiLetter(static_cast<uint8_t>(text[pos])))
                {
                    ++pos;
                }
            }
            else if (IsAsciiDigit(next))
            {
                ++pos;
            }
            else
            {
                while (pos < text.size())
                {
                    const auto c = static_cast<uint8_t>(text[pos]);
                    if (IsAsciiWhitespace(c) || IsAsciiLetter(c) || IsAsciiDigit(c))
                    {
                        break;
                    }
                    ++pos;
                }
                while (pos < text.size() && (text[pos] == '\r' || text[pos] == '\n'))
                {
                    ++pos;
                }
            }
            pieces.emplace_back(text.substr(start, pos - start));
            continue;
        }

        if (IsAsciiLetter(current))
        {
            const size_t start = pos;
            while (pos < text.size() && IsAsciiLetter(static_cast<uint8_t>(text[pos])))
            {
                ++pos;
            }
            pieces.emplace_back(text.substr(start, pos - start));
        }
        else if (IsAsciiDigit(current))
        {
            pieces.emplace_back(text.substr(pos, 1));
            ++pos;
        }
        else if (current == '\r' || current == '\n')
        {
            const size_t start = pos;
            while (pos < text.size() && IsAsciiWhitespace(static_cast<uint8_t>(text[pos])))
            {
                if (pos > start && text[pos] != '\r' && text[pos] != '\n')
                {
                    break;
                }
                ++pos;
            }
            pieces.emplace_back(text.substr(start, pos - start));
        }
        else if (IsAsciiWhitespace(current))
        {
            const size_t start = pos;
            while (pos < text.size() && IsAsciiWhitespace(static_cast<uint8_t>(text[pos])))
            {
                if (text[pos] == '\r' || text[pos] == '\n')
                {
                    break;
                }
                ++pos;
            }
            pieces.emplace_back(text.substr(start, pos - start));
        }
        else
        {
            const size_t start = pos;
            while (pos < text.size())
            {
                const auto c = static_cast<uint8_t>(text[pos]);
                if (IsAsciiWhitespace(c) || IsAsciiLetter(c) || IsAsciiDigit(c))
                {
                    break;
                }
                ++pos;
            }
            while (pos < text.size() && (text[pos] == '\r' || text[pos] == '\n'))
            {
                ++pos;
            }
            pieces.emplace_back(text.substr(start, pos - start));
        }
    }
    return pieces;
}

std::vector<std::string> ApplyByteLevelBpe(std::string_view piece, const std::array<std::string, 256>& byte_encoder,
                                           const std::unordered_map<std::string, int32_t>& bpe_ranks)
{
    std::vector<std::string> tokens;
    tokens.reserve(piece.size());
    for (const char c : piece)
    {
        tokens.push_back(byte_encoder[static_cast<uint8_t>(c)]);
    }

    while (tokens.size() > 1)
    {
        auto best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = tokens.size();
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
        {
            const auto rank = bpe_ranks.find(BpePairKey(tokens[i], tokens[i + 1]));
            if (rank != bpe_ranks.end() && rank->second < best_rank)
            {
                best_rank = rank->second;
                best_index = i;
            }
        }
        if (best_index == tokens.size())
        {
            break;
        }
        tokens[best_index] += tokens[best_index + 1];
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    return tokens;
}

}  // namespace

int64_t Tokenizer::TokenId(const std::string& token) const
{
    const auto it = token_to_id_.find(token);
    if (it == token_to_id_.end())
    {
        return -1;
    }
    return it->second;
}

std::vector<int64_t> Tokenizer::Encode(const std::string& text, bool add_special_tokens) const
{
    if (bpe_ranks_.empty())
    {
        throw std::runtime_error("Tokenizer encoding is only available for byte-level BPE tokenizer.json format");
    }
    (void)add_special_tokens;

    std::vector<int64_t> ids;
    const auto encode_segment = [&](std::string_view segment)
    {
        for (const auto& piece : SplitForByteLevelBpe(segment))
        {
            for (const auto& token : ApplyByteLevelBpe(piece, byte_encoder_, bpe_ranks_))
            {
                const auto id = token_to_id_.find(token);
                if (id == token_to_id_.end())
                {
                    throw std::runtime_error("tokenizer vocab is missing BPE token: " + token);
                }
                ids.push_back(id->second);
            }
        }
    };

    size_t segment_start = 0;
    size_t pos = 0;
    while (pos < text.size())
    {
        const std::string* matched_special = nullptr;
        for (const auto& special : special_tokens_)
        {
            if (StartsWith(std::string_view(text).substr(pos), special))
            {
                matched_special = &special;
                break;
            }
        }
        if (matched_special == nullptr)
        {
            ++pos;
            continue;
        }

        encode_segment(std::string_view(text).substr(segment_start, pos - segment_start));
        ids.push_back(token_to_id_.at(*matched_special));
        pos += matched_special->size();
        segment_start = pos;
    }
    encode_segment(std::string_view(text).substr(segment_start));
    return ids;
}

std::string Tokenizer::CleanToken(int64_t id) const
{
    const std::string& raw = Token(id);
    if (raw.empty())
    {
        return {};
    }
    constexpr std::string_view marker = "\xE2\x96\x81";
    if (StartsWith(raw, marker))
    {
        return std::string(raw.substr(marker.size()));
    }
    if (StartsWith(raw, "##"))
    {
        return std::string(raw.substr(2));
    }
    return raw;
}

std::string Tokenizer::Decode(const std::vector<int64_t>& ids, bool skip_special_tokens, bool strip_lang_tags) const
{
    if (decode_mode_ == DecodeMode::WhisperByteBpe)
    {
        std::string text;
        for (const int64_t id : ids)
        {
            if (id < 0 || id >= kWhisperTextTokenLimit)
            {
                continue;
            }
            text += id_to_token_[static_cast<size_t>(id)];
        }
        return text;
    }

    std::string text;
    for (int64_t id : ids)
    {
        std::string_view piece = Token(id);
        if (piece.empty() || (skip_special_tokens && piece.front() == '<'))
        {
            continue;
        }
        if (StartsWith(piece, "##"))
        {
            text += piece.substr(2);
        }
        else if (StartsWith(piece, "\xE2\x96\x81"))
        {
            if (!text.empty())
            {
                text += ' ';
            }
            text += piece.substr(3);
        }
        else
        {
            text += piece;
        }
    }
    if (strip_lang_tags)
    {
        text = StripLangTags(text);
    }
    const auto first = text.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\n\r");
    return text.substr(first, last - first + 1);
}

std::string Tokenizer::LanguageCode(int64_t lang_token) const
{
    const auto& token = Token(lang_token);
    if ((token.size() == 6 || token.size() == 7) && token.starts_with("<|") && token.ends_with("|>"))
        return token.substr(2, token.size() - 4);
    const int64_t index = lang_token - kWhisperLangFirst;
    if (index < 0 || index >= static_cast<int64_t>(lang_codes_.size()))
    {
        return {};
    }
    return lang_codes_[static_cast<size_t>(index)];
}

}  // namespace din::io
