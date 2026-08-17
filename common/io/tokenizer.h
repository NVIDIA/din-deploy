// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace din::io
{

enum class TokenizerFormat
{
    Json,
    Vocab,
    WhisperJson,
};

inline constexpr int64_t kWhisperEndOfText = 50257;
inline constexpr int64_t kWhisperStartOfTranscript = 50258;
inline constexpr int64_t kWhisperLangFirst = 50259;
inline constexpr int64_t kWhisperLangLast = 50357;
inline constexpr int64_t kWhisperTranslate = 50358;
inline constexpr int64_t kWhisperTranscribe = 50359;
inline constexpr int64_t kWhisperStartOfPrev = 50361;
inline constexpr int64_t kWhisperNoTimestamps = 50363;
inline constexpr int64_t kWhisperTimestampFirst = 50364;
inline constexpr int64_t kWhisperTextTokenLimit = kWhisperEndOfText;

class Tokenizer
{
public:
    explicit Tokenizer(const std::string& path, TokenizerFormat format);

    [[nodiscard]] const std::string& Token(int64_t id) const;
    [[nodiscard]] int64_t TokenId(const std::string& token) const;
    [[nodiscard]] std::vector<int64_t> Encode(const std::string& text, bool add_special_tokens = true) const;
    [[nodiscard]] std::string CleanToken(int64_t id) const;
    [[nodiscard]] std::string Decode(const std::vector<int64_t>& ids, bool skip_special_tokens = true,
                                     bool strip_lang_tags = false) const;

    [[nodiscard]] std::string LanguageCode(int64_t lang_token) const;

private:
    enum class DecodeMode
    {
        Pieces,
        WhisperByteBpe,
    };

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::unordered_map<std::string, int32_t> bpe_ranks_;
    std::array<std::string, 256> byte_encoder_;
    std::vector<std::string> special_tokens_;
    std::vector<std::string> lang_codes_;
    DecodeMode decode_mode_ = DecodeMode::Pieces;
};

}  // namespace din::io
