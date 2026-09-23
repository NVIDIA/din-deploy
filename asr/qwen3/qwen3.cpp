// SPDX-License-Identifier: Apache-2.0
#include "qwen3.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <sstream>

#include "detail/runtime.h"

namespace din::asr::qwen3::detail
{
constexpr int kMaxSamples = 1205 * kRate;
constexpr int kPrefillBlock = 512;

inline int ResolveChunkSeconds(int seconds)
{
    if (seconds == 0)
        return 1200;
    // A target inside the search radius can repeatedly split silence into single samples.
    if (seconds <= 5 || seconds > 1200)
        throw std::invalid_argument("max-chunk-seconds must be 0 (auto) or 6..1200");
    return seconds;
}

struct AudioChunk
{
    size_t begin;
    size_t end;
};

inline float WindowEnergy(std::span<const float> samples, size_t begin)
{
    // Independent FP32 reductions avoid cancellation drift across quiet windows.
    std::array<float, 8> sums{};
    for (size_t i = 0; i < 1600; i += 8)
        for (size_t j = 0; j < 8; ++j)
            sums[j] += std::abs(samples[begin + i + j]);
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

// Qwen3-ASR split_audio_into_chunks: quietest 100 ms within +/-5 seconds,
// then the quietest sample inside that window. Keep the first minimum on ties.
inline std::vector<AudioChunk> SplitAudio(std::span<const float> samples, size_t target_seconds = 1200)
{
    const size_t target = target_seconds * 16000;
    constexpr size_t expand = 5 * 16000, window = 1600;
    std::vector<AudioChunk> chunks;
    size_t start = 0;
    while (samples.size() - start > target)
    {
        const auto cut = start + target;
        const auto left = cut > expand ? std::max(start, cut - expand) : start;
        const auto right = std::min(samples.size(), cut + expand);
        size_t boundary = cut;
        if (right - left > window)
        {
            float best = WindowEnergy(samples, left);
            size_t minimum = left;
            for (size_t i = left + 1; i + window <= right; ++i)
            {
                const float sum = WindowEnergy(samples, i);
                if (sum < best)
                {
                    best = sum;
                    minimum = i;
                }
            }
            boundary = minimum;
            for (size_t i = minimum + 1; i < minimum + window; ++i)
                if (std::abs(samples[i]) < std::abs(samples[boundary]))
                    boundary = i;
        }
        boundary = std::clamp(boundary, start + 1, samples.size());
        chunks.push_back({start, boundary});
        start = boundary;
    }
    if (start < samples.size())
        chunks.push_back({start, samples.size()});
    return chunks;
}
}  // namespace din::asr::qwen3::detail

namespace din::asr::qwen3::detail
{
std::string Trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \r\n\t");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \r\n\t") - first + 1);
}

// Official detect_and_fix_repetitions operates on Unicode characters, not UTF-8 bytes.
static std::string FixRepetitions(const std::string& text)
{
    std::vector<std::string_view> chars, filtered;
    for (size_t i = 0; i < text.size();)
    {
        const auto c = static_cast<unsigned char>(text[i]);
        const size_t length = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        chars.emplace_back(text.data() + i, std::min(length, text.size() - i));
        i += length;
    }
    for (size_t i = 0; i < chars.size();)
    {
        size_t end = i + 1;
        while (end < chars.size() && chars[end] == chars[i])
            ++end;
        filtered.insert(filtered.end(), chars.begin() + i, chars.begin() + (end - i > 20 ? i + 1 : end));
        i = end;
    }
    std::string result;
    for (size_t i = 0; i < filtered.size();)
    {
        size_t advance = 1, keep = 1;
        if (filtered.size() - i >= 40)
            for (size_t length = 1; length <= 20 && i + length * 20 <= filtered.size(); ++length)
            {
                size_t end = i + length;
                while (end + length <= filtered.size() &&
                       std::equal(filtered.begin() + i, filtered.begin() + i + length, filtered.begin() + end))
                    end += length;
                if ((end - i) / length >= 20)
                {
                    keep = length;
                    advance = end - i;
                    break;
                }
            }
        for (size_t j = 0; j < keep; ++j)
            result += filtered[i + j];
        i += advance;
    }
    return result;
}

std::pair<std::string, std::string> ParseOutput(const std::string& raw, const std::string& forced_language = {})
{
    const auto text = FixRepetitions(Trim(raw));
    if (text.empty())
        return {};
    if (!forced_language.empty())
        return {forced_language, text};
    const auto marker = text.find("<asr_text>");
    if (marker == std::string::npos)
        return {{}, Trim(text)};
    auto meta = text.substr(0, marker);
    std::transform(meta.begin(), meta.end(), meta.begin(),
                   [](unsigned char c)
                   {
                       return std::tolower(c);
                   });
    const auto transcript = Trim(text.substr(marker + 10));
    if (meta.find("language none") != std::string::npos)
        return {{}, transcript};
    std::istringstream lines(meta);
    std::string line;
    while (std::getline(lines, line))
    {
        line = Trim(line);
        if (line.starts_with("language "))
        {
            auto language = Trim(line.substr(9));
            if (!language.empty())
                language[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(language[0])));
            return {language, transcript};
        }
    }
    return {{}, transcript};
}

std::string AsrLanguage(const std::string& language, const nlohmann::json& languages)
{
    const auto hint = LowerLanguage(language);
    for (const auto& [key, value] : languages.items())
        if (LowerLanguage(key) == hint)
            return key;
    throw std::invalid_argument("Unknown ASR language hint; use an upstream language code/name or auto");
}

}  // namespace din::asr::qwen3::detail

namespace din::asr::qwen3
{
using namespace detail;
struct Qwen3Pipeline::Impl
{
    Qwen3Config config;
    Runtime runtime;
    AudioModel asr;
    std::unique_ptr<OrtRunner> text, prefill_session;
    int64_t capacity = 0;
    std::vector<int64_t> eos;
    std::vector<Ort::Value> cache;
    std::unique_ptr<TextInputs> step, prefill;
    Ort::Value logits{nullptr};
    std::unique_ptr<Buffer<int64_t>> next_token;
    std::unique_ptr<Ort::IoBinding> decode_binding, prefill_binding;
    Ort::RunOptions decode_options;

    explicit Impl(Qwen3Config cfg)
        : config(std::move(cfg))
        , runtime(config.provider, config.ep_cache_dir, config.ep_context_dir, config.progress)
        , asr(runtime, config.model_dir, "asr")
    {
        if (config.max_new_tokens <= 0)
            throw std::runtime_error("max-new-tokens must be positive");
        config.max_chunk_seconds = detail::ResolveChunkSeconds(config.max_chunk_seconds);
        const auto& meta = asr.metadata;
        if (meta.at("prefill_block") != kPrefillBlock || meta.at("cache_capacity").get<int64_t>() < kPrefillBlock ||
            meta.at("cache_capacity").get<int64_t>() % kPrefillBlock)
            throw std::runtime_error("Unsupported Qwen3 cache geometry");
        eos = asr.metadata["eos_token_ids"].get<std::vector<int64_t>>();
        config.lang_id = detail::AsrLanguage(config.lang_id, asr.native["languages"]);
        const auto cap = meta["cache_capacity"].get<int64_t>();
        const auto decode_shape = TextShape(1, asr.hidden, cap) + ",logits_index:1";
        const auto prefill_shape = TextShape(kPrefillBlock, asr.hidden, cap) + ",logits_index:1";
        text = runtime.Runner(config.model_dir, "decoder", decode_shape, decode_shape, decode_shape);
        prefill_session =
            runtime.Runner(config.model_dir, "decoder", prefill_shape, prefill_shape, prefill_shape, "_prefill");
        if (text->HasDeviceIo())
            decode_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        runtime.LoadMel(config.model_dir);
    }

    void PrepareCache()
    {
        if (!cache.empty())
            return;
        capacity = asr.metadata.at("cache_capacity");
        const auto& c = asr.metadata["text_config"];
        const std::vector<int64_t> shape{1, c["num_key_value_heads"], capacity, c["head_dim"]};
        for (int i = 0; i < 2 * c["num_hidden_layers"].get<int>(); ++i)
            cache.push_back(TensorValue(*text, shape, asr.dtype));
        step = std::make_unique<TextInputs>(*text, 1, asr.hidden, capacity, asr.dtype);
        prefill = std::make_unique<TextInputs>(*prefill_session, kPrefillBlock, asr.hidden, capacity, asr.dtype);
        logits = TensorValue(*text, {1, c["vocab_size"]}, asr.dtype);
        next_token = std::make_unique<Buffer<int64_t>>(*text, std::vector<int64_t>{1}, text->HasDeviceIo());
        decode_binding = std::make_unique<Ort::IoBinding>(text->session);
        prefill_binding = std::make_unique<Ort::IoBinding>(prefill_session->session);
        step->Bind(*decode_binding, true);
        prefill->Bind(*prefill_binding, true);
        BindDecode(*decode_binding);
        BindDecode(*prefill_binding);
    }

    void BindDecode(Ort::IoBinding& binding)
    {
        for (size_t i = 0; i < cache.size(); ++i)
        {
            binding.BindInput(("past_" + std::to_string(i)).c_str(), cache[i]);
            binding.BindOutput(("present_" + std::to_string(i)).c_str(), cache[i]);
        }
        binding.BindOutput("logits", logits);
        binding.BindOutput("next_token", next_token->BindingValue());
    }

    TranscriptionResult TranscribeChunk(std::span<const float> audio)
    {
        if (audio.empty() || audio.size() > kMaxSamples)
            throw std::runtime_error("Expected nonempty mono 16 kHz audio, at most 1205 seconds per chunk");
        auto features = runtime.Features(audio);
        const int64_t frames = features.size() / 128;
        auto encoded = asr.Encode(features, frames);
        std::vector<int64_t> ids = asr.native["prefixes"][config.lang_id];
        ids.insert(ids.end(), encoded.tokens, asr.audio_id);
        if (!asr.native.contains("suffixes"))
            throw std::runtime_error("Re-export native prompt assets with --only mel for official language forcing");
        Append(ids, asr.native["suffixes"][config.lang_id].get<std::vector<int64_t>>());
        PrepareCache();
        if (ids.size() + config.max_new_tokens > static_cast<size_t>(capacity))
            throw std::runtime_error("Prompt and generation budget exceed the exported KV capacity");
        // Clear on the shared CUDA stream. Unused NaN cache values can poison attention even when masked.
        for (auto& value : cache)
            ZeroTensor(*text, value);
        int64_t audio_offset = 0;
        for (size_t offset = 0; offset < ids.size(); offset += kPrefillBlock)
        {
            din::common::nvtx_scoped_range range{"qwen3.prefill"};
            const auto block =
                std::span<const int64_t>(ids).subspan(offset, std::min<size_t>(kPrefillBlock, ids.size() - offset));
            prefill->Fill(block, offset, asr.audio_id, &encoded, audio_offset);
            audio_offset += std::count(block.begin(), block.end(), asr.audio_id);
            prefill_session->session.Run(Ort::RunOptions{}, *prefill_binding);
            prefill_binding->SynchronizeOutputs();
        }
        int64_t position = ids.size();
        TranscriptionResult result;
        for (int count = 0; count < config.max_new_tokens; ++count)
        {
            din::common::nvtx_scoped_range range{"qwen3.decode_step"};
            next_token->CopyAsyncToHostWithNotification().Sync();
            const auto token = next_token->HostData()[0];
            result.tokens.push_back(token);
            if (std::find(eos.begin(), eos.end(), token) != eos.end())
            {
                result.reached_eos = true;
                break;
            }
            if (position >= capacity || count + 1 == config.max_new_tokens)
                break;
            step->Fill(std::span(&token, 1), position++, asr.audio_id, nullptr);
            text->session.Run(decode_options, *decode_binding);
        }
        auto text_tokens = result.tokens;
        if (result.reached_eos)
            text_tokens.pop_back();
        const auto parsed = detail::ParseOutput(asr.tokenizer->Decode(text_tokens, false),
                                                asr.native["languages"][config.lang_id].get<std::string>());
        result.language = parsed.first;
        result.text = parsed.second;
        return result;
    }

    TranscriptionResult Transcribe(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.transcribe"};
        const auto start = std::chrono::steady_clock::now();
        din::io::Audio normalized;
        const auto* source = &NormalizeAudio(audio, normalized);
        if (config.progress)
            config.progress({din::common::ProgressStage::Transcribing, "Transcribing", 0, audio.Duration()});
        TranscriptionResult result;
        result.reached_eos = true;
        std::string previous_language;
        const int64_t prompt =
            asr.native["prefixes"][config.lang_id].size() + asr.native["suffixes"][config.lang_id].size();
        const int64_t available = asr.metadata["cache_capacity"].get<int64_t>() - prompt - config.max_new_tokens;
        const int seconds = static_cast<int>(available / 13);
        if (seconds < 1)
            throw std::invalid_argument(
                "Generation budget leaves no room for audio; increase cache-capacity or reduce max-new-tokens");
        // Preserve upstream boundaries when possible; reserve the +5s quiet search margin.
        const int target = std::min(config.max_chunk_seconds, seconds - 5);
        if (source->samples.size() > static_cast<size_t>(seconds) * kRate && target < 6)
            throw std::invalid_argument("KV capacity is too small for long-form quiet-boundary splitting");
        const auto chunks = source->samples.size() <= static_cast<size_t>(seconds) * kRate && target < 6
                                ? std::vector<detail::AudioChunk>{{0, source->samples.size()}}
                                : detail::SplitAudio(source->samples, std::max(6, target));
        for (const auto chunk : chunks)
        {
            auto part = TranscribeChunk(std::span(source->samples).subspan(chunk.begin, chunk.end - chunk.begin));
            result.text += part.text;  // Upstream joins literally, without overlap or inserted separators.
            Append(result.tokens, part.tokens);
            if (!part.language.empty() && part.language != previous_language)
            {
                if (!result.language.empty())
                    result.language += ",";
                result.language += part.language;
                previous_language = part.language;
            }
            result.segments.push_back({std::move(part.text), std::move(part.language), chunk.begin, chunk.end});
            ++result.chunks_processed;
            result.reached_eos = result.reached_eos && part.reached_eos;
            if (config.progress)
                config.progress(
                    {din::common::ProgressStage::Transcribing,
                     "Completed chunk " + std::to_string(result.chunks_processed),
                     chunk.end == source->samples.size() ? audio.Duration() : static_cast<float>(chunk.end) / kRate,
                     audio.Duration()});
        }
        result.audio_seconds = static_cast<float>(audio.Duration());
        result.transcribe_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        return result;
    }
};

Qwen3Pipeline::Qwen3Pipeline(Qwen3Config config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}
Qwen3Pipeline::~Qwen3Pipeline() = default;
TranscriptionResult Qwen3Pipeline::Transcribe(const din::io::Audio& audio)
{
    return impl_->Transcribe(audio);
}
TranscriptionResult Qwen3Pipeline::TranscribeFile(const std::filesystem::path& path)
{
    if (impl_->config.progress)
        impl_->config.progress({din::common::ProgressStage::DecodingAudio, path.filename().string()});
    return Transcribe(din::io::LoadAudio(path, kRate));
}
}  // namespace din::asr::qwen3
