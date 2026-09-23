// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdlib>
#include <unordered_map>

#include "runtime.h"
#include "text.h"
#include "unicode_regex.h"

namespace din::asr::qwen3::detail
{
// Nagisa preprocessing, dictionary features and BMES decoding; neural inference stays in ORT.
class JapaneseTokenizer
{
    using Vocabulary = std::unordered_map<std::string, int64_t>;
    Vocabulary unigrams_, bigrams_, words_;
    int window_;
    int64_t padding_word_;
    std::array<std::array<float, 6>, 6> transitions_;
    Ort::Session session_{nullptr};

    static std::string Utf8(int32_t code)
    {
        utf8proc_uint8_t data[4];
        const auto size = utf8proc_encode_char(code, data);
        return {reinterpret_cast<char*>(data), static_cast<size_t>(size)};
    }
    static int64_t Lookup(const Vocabulary& vocabulary, const std::string& text)
    {
        const auto found = vocabulary.find(text);
        return found == vocabulary.end() ? vocabulary.at("oov") : found->second;
    }

public:
    JapaneseTokenizer(Ort::Env& env, const std::filesystem::path& dir)
    {
        if (!std::filesystem::exists(dir / "japanese.json"))
            throw std::runtime_error("Japanese word alignment requires --task aligner --only japanese export");
        const auto meta = ReadJson(dir / "japanese.json");
        unigrams_ = meta.at("unigrams").get<Vocabulary>();
        bigrams_ = meta.at("bigrams").get<Vocabulary>();
        words_ = meta.at("words").get<Vocabulary>();
        window_ = meta.at("window");
        padding_word_ = meta.at("padding_word");
        transitions_ = meta.at("transitions").get<decltype(transitions_)>();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        session_ = Ort::Session(env, (dir / "japanese.onnx").c_str(), options);
    }

    std::vector<std::string> Words(std::string text)
    {
        if (!ValidUtf8(text))
            throw std::invalid_argument("Japanese alignment requires valid UTF-8 text");
        static const din::io::UnicodeRegex leading(R"(^[\s\x{1c}-\x{1f}]+)"), trailing(R"([\s\x{1c}-\x{1f}]+$)");
        const auto head = leading.FindAll(text);
        if (!head.empty())
            text.erase(0, head[0].size());
        const auto tail = trailing.FindAll(text);
        if (!tail.empty())
            text.resize(text.size() - tail[0].size());
        utf8proc_uint8_t* normalized = nullptr;
        const auto length =
            utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()), text.size(), &normalized,
                         static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPAT | UTF8PROC_COMPOSE));
        const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owner(normalized, std::free);
        if (length < 0)
            throw std::invalid_argument(utf8proc_errmsg(length));
        std::vector<std::string> characters, lower;
        std::vector<int64_t> types;
        for (size_t i = 0; i < static_cast<size_t>(length);)
        {
            int32_t code;
            i += utf8proc_iterate(normalized + i, length - i, &code);
            if (code == 0x130)
                code = 'I';
            if (code == ' ')
                code = 0x3000;
            characters.push_back(Utf8(code));
            code = utf8proc_tolower(code);
            lower.push_back(Utf8(code));
            types.push_back(code >= 0x3040 && code <= 0x309f   ? 0
                            : code >= 0x30a1 && code <= 0x30fa ? 1
                            : code >= 0x4e00 && code <= 0x9fa5 ? 2
                            : code >= 'a' && code <= 'z'       ? 3
                            : code >= '0' && code <= '9'       ? 4
                                                               : 5);
        }
        const int64_t count = characters.size();
        if (!count)
            return {};
        std::array<std::vector<int64_t>, 5> features;
        for (size_t i = 0; i < 3; ++i)
            features[i].resize(count * window_, i == 2 ? 6 : 1);
        for (size_t i = 3; i < 5; ++i)
            features[i].resize(count * 8, padding_word_);
        for (int64_t i = 0; i < count; ++i)
        {
            for (int j = 0; j < window_; ++j)
            {
                const auto at = i + j - window_ / 2;
                if (at < 0 || at >= count)
                    continue;
                features[0][i * window_ + j] = Lookup(unigrams_, lower[at]);
                features[1][i * window_ + j] = Lookup(bigrams_, lower[at] + (at + 1 < count ? lower[at + 1] : "<E>"));
                features[2][i * window_ + j] = types[at];
            }
            for (int direction = 0; direction < 2; ++direction)
            {
                std::string word;
                int matches = 0;
                for (int j = 0; j < 8; ++j)
                {
                    const auto at = direction ? i - j : i + j;
                    if (at < 0 || at >= count)
                        break;
                    word = direction ? lower[at] + word : word + lower[at];
                    if (const auto found = words_.find(word); found != words_.end())
                        features[3 + direction][i * 8 + matches++] = found->second;
                }
                if (!matches)
                    features[3 + direction][i * 8] = words_.at("oov");
            }
        }
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        for (size_t i = 0; i < features.size(); ++i)
        {
            const int64_t shape[]{count, i < 3 ? window_ : 8};
            inputs.push_back(Ort::Value::CreateTensor(memory, features[i].data(), features[i].size(), shape, 2));
        }
        const char* names[]{"unigrams", "bigrams", "types", "word_starts", "word_ends"};
        const char* output_name = "emissions";
        auto outputs = session_.Run(Ort::RunOptions{}, names, inputs.data(), inputs.size(), &output_name, 1);
        const auto* emissions = outputs[0].GetTensorData<float>();
        std::vector<std::array<int, 6>> parents(count);
        std::array<float, 6> scores;
        scores.fill(-1e10f);
        scores[4] = 0;
        for (int64_t i = 0; i < count; ++i)
        {
            std::array<float, 6> next;
            for (int to = 0; to < 6; ++to)
            {
                int best = 0;
                for (int from = 1; from < 6; ++from)
                    if (scores[from] + transitions_[to][from] > scores[best] + transitions_[to][best])
                        best = from;
                parents[i][to] = best;
                next[to] = scores[best] + transitions_[to][best] + emissions[i * 6 + to];
            }
            // A common offset preserves the best path while keeping long sequences in FP32 range.
            const float maximum = *std::max_element(next.begin(), next.end());
            for (int j = 0; j < 6; ++j)
                scores[j] = next[j] - maximum;
        }
        int tag = 0;
        for (int i = 1; i < 6; ++i)
            if (scores[i] + transitions_[5][i] > scores[tag] + transitions_[5][tag])
                tag = i;
        std::vector<int> tags(count);
        for (int64_t i = count; i-- > 0;)
        {
            tags[i] = tag;
            tag = parents[i][tag];
        }
        static const din::io::UnicodeRegex kept(R"([\p{L}\p{N}']+)");
        std::vector<std::string> result;
        std::string word;
        auto emit = [&]
        {
            std::string cleaned;
            for (const auto& part : kept.FindAll(word))
                cleaned += part;
            if (!cleaned.empty())
                result.push_back(std::move(cleaned));
            word.clear();
        };
        for (int64_t i = 0; i < count; ++i)
        {
            if (tags[i] == 3)
                emit();
            word += characters[i];
            if (tags[i] == 2 || tags[i] == 3)
                emit();
        }
        emit();
        return result;
    }
};
}  // namespace din::asr::qwen3::detail
