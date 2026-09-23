// SPDX-License-Identifier: Apache-2.0
#include "forced_aligner.h"

#include "detail/runtime.h"
#include "unicode_regex.h"

namespace din::asr::qwen3::detail
{
std::vector<std::string> AlignmentUnits(const std::string& text, const std::string& language)
{
    static constexpr std::pair<std::string_view, std::string_view> languages[] = {
        {"zh", "chinese"}, {"yue", "cantonese"}, {"en", "english"}, {"de", "german"},
        {"es", "spanish"}, {"fr", "french"},     {"it", "italian"}, {"pt", "portuguese"},
        {"ru", "russian"}, {"ko", "korean"},     {"ja", "japanese"}};
    const auto name = LowerLanguage(language);
    const auto found = std::find_if(std::begin(languages), std::end(languages),
                                    [&](const auto& entry)
                                    {
                                        return name == entry.first || name == entry.second;
                                    });
    if (found == std::end(languages))
        throw std::invalid_argument("Forced alignment supports zh, yue, en, de, es, fr, it, pt, ru, ko and ja");

    // HF keeps Unicode letters/numbers and ASCII apostrophes, dropping punctuation and marks.
    static const din::io::UnicodeRegex kept(R"([\p{L}\p{N}'\s\x{1c}-\x{1f}]+)");
    std::string cleaned;
    for (const auto& part : kept.FindAll(text))
        cleaned += part;
    static const std::string cjk = R"(\x{4e00}-\x{9fff}\x{3400}-\x{4dbf}\x{20000}-\x{2a6df}\x{2a700}-\x{2b73f})"
                                   R"(\x{2b740}-\x{2b81f}\x{2b820}-\x{2ceaf}\x{f900}-\x{faff}\x{2f800}-\x{2fa1f})";
    static const din::io::UnicodeRegex words("[" + cjk + "]|[^" + cjk + R"(\s\x{1c}-\x{1f}]+)");
    // HF's unscored Korean LTokenizer splits on whitespace before cleaning each unit.
    static const din::io::UnicodeRegex korean(R"([^\s\x{1c}-\x{1f}]+)");
    // Japanese character timestamps avoid a separate Nagisa word-segmentation runtime.
    static const std::string kana = R"(\x{3040}-\x{30ff}\x{31f0}-\x{31ff}\x{ff66}-\x{ff9f})";
    static const din::io::UnicodeRegex japanese("[" + cjk + kana + "]|[^" + cjk + kana + R"(\s\x{1c}-\x{1f}]+)");
    if (found->first == "ja")
        return japanese.FindAll(cleaned);
    if (found->first == "ko")
        return korean.FindAll(cleaned);
    return words.FindAll(cleaned);
}

// HF's nondecreasing subsequence repair, including its tie rules.
std::vector<int> FixTimestamps(const std::vector<int>& data)
{
    const int n = static_cast<int>(data.size());
    if (n == 0)
        return {};
    std::vector<int> dp(n, 1), parent(n, -1), result = data;
    std::vector<bool> normal(n, false);
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (data[j] <= data[i] && dp[j] + 1 > dp[i])
            {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
    int i = static_cast<int>(std::max_element(dp.begin(), dp.end()) - dp.begin());
    for (; i >= 0; i = parent[i])
        normal[i] = true;
    for (int begin = 0; begin < n;)
    {
        if (normal[begin])
        {
            ++begin;
            continue;
        }
        int end = begin;
        while (end < n && !normal[end])
            ++end;
        for (int k = begin; k < end; ++k)
        {
            if (begin == 0)
                result[k] = result[end];
            else if (end == n)
                result[k] = result[begin - 1];
            else if (end - begin <= 2)
                result[k] = k - begin + 1 <= end - k ? result[begin - 1] : result[end];
            else
                result[k] = static_cast<int>(result[begin - 1] + (result[end] - result[begin - 1]) *
                                                                     float(k - begin + 1) / (end - begin + 1));
        }
        begin = end;
    }
    return result;
}

struct AlignmentEngine
{
    AudioModel model;
    std::unique_ptr<OrtRunner> text;
    AlignmentEngine(Runtime& runtime, const std::filesystem::path& dir);
    std::vector<WordTimestamp> Align(const std::vector<float>& features, const std::string& transcript,
                                     const std::string& language);
};

AlignmentEngine::AlignmentEngine(Runtime& runtime, const std::filesystem::path& dir)
    : model(runtime, dir, "aligner")
{
    if (!model.metadata.value("timestamp_bins", false))
        throw std::runtime_error("Re-export the aligner with --only aligner for GPU timestamp selection");
    const auto shape = [&](int64_t seq, int slots)
    {
        return TextShape(seq, model.hidden, seq) + ",timestamp_indices:" + std::to_string(slots);
    };
    text = runtime.Runner(dir, "aligner", shape(4, 2), shape(128, 32), shape(8192, 4096));
}
std::vector<WordTimestamp> AlignmentEngine::Align(const std::vector<float>& features, const std::string& transcript,
                                                  const std::string& language)
{
    din::common::nvtx_scoped_range range{"qwen3.align"};
    if (transcript.empty())
        return {};
    const int64_t frames = features.size() / 128;
    std::vector<WordTimestamp> result;
    const auto words = detail::AlignmentUnits(transcript, language);
    if (words.empty())
        return {};
    if (words.size() > 2048)
        throw std::runtime_error("Native alignment is limited to 2048 alignment units per chunk");
    auto audio = [&]
    {
        din::common::nvtx_scoped_range range{"qwen3.align_encoder"};
        return model.Encode(features, frames);
    }();
    std::vector<int64_t> ids = model.native["audio_start"];
    ids.insert(ids.end(), audio.tokens, model.audio_id);
    Append(ids, model.native["audio_end"].get<std::vector<int64_t>>());
    std::vector<int64_t> slots;
    const int64_t timestamp_id = model.metadata["timestamp_token_id"];
    for (const auto& word : words)
    {
        Append(ids, model.tokenizer->Encode(word, false));
        for (int i = 0; i < 2; ++i)
        {
            slots.push_back(ids.size());
            ids.push_back(timestamp_id);
        }
    }
    const auto seq = static_cast<int64_t>(ids.size());
    if (seq > 8192)
        throw std::runtime_error("Alignment exceeds the native context limit");
    TextInputs input(*text, seq, model.hidden, seq, model.dtype);
    input.Fill(ids, 0, model.audio_id, &audio);
    const int64_t labels = model.metadata["num_labels"];
    Buffer<int64_t> indices(*text, {static_cast<int64_t>(slots.size())}, text->HasDeviceIo());
    auto logits = TensorValue(*text, {1, static_cast<int64_t>(slots.size()), labels}, model.dtype);
    Buffer<int64_t> output(*text, {1, static_cast<int64_t>(slots.size())}, text->HasDeviceIo());
    std::copy(slots.begin(), slots.end(), indices.HostData());
    indices.CopyAsyncToDevice();
    Ort::IoBinding binding(text->session);
    input.Bind(binding);
    binding.BindInput("timestamp_indices", indices.BindingValue());
    binding.BindOutput("timestamp_logits", logits);
    binding.BindOutput("timestamp_bins", output.BindingValue());
    {
        din::common::nvtx_scoped_range range{"qwen3.align_inference"};
        text->session.Run(Ort::RunOptions{}, binding);
    }
    {
        din::common::nvtx_scoped_range range{"qwen3.align_download"};
        output.CopyAsyncToHostWithNotification().Sync();
    }
    din::common::nvtx_scoped_range postprocess{"qwen3.align_postprocess"};
    const float timestamp_scale = model.metadata["timestamp_segment_time"];
    std::vector<int> raw;
    for (size_t i = 0; i < slots.size(); ++i)
    {
        const auto best = output.HostData()[i];
        if (best < 0 || best >= labels)
            throw std::runtime_error("Invalid timestamp bin");
        raw.push_back(static_cast<int>(best * timestamp_scale));
    }
    const auto times = detail::FixTimestamps(raw);
    for (size_t i = 0; i < words.size(); ++i)
        result.push_back({words[i], times[2 * i] / 1000.f, times[2 * i + 1] / 1000.f});
    return result;
}
}  // namespace din::asr::qwen3::detail

namespace din::asr::qwen3
{
using namespace detail;
struct Qwen3ForcedAligner::Impl
{
    ForcedAlignerConfig config;
    Runtime runtime;
    AlignmentEngine engine;
    explicit Impl(ForcedAlignerConfig cfg)
        : config(std::move(cfg))
        , runtime(config.provider, config.ep_cache_dir, config.ep_context_dir, config.progress)
        , engine(runtime, config.model_dir)
    {
        runtime.LoadMel(config.model_dir);
    }
    std::vector<WordTimestamp> AlignAudio(const din::io::Audio& audio, const std::string& text,
                                          const std::string& language)
    {
        din::io::Audio normalized;
        const auto& source = NormalizeAudio(audio, normalized);
        if (source.samples.size() > 180 * kRate)
            throw std::invalid_argument("Standalone alignment accepts up to 180 seconds; supply audio/text segments");
        if (config.progress)
            config.progress({din::common::ProgressStage::Aligning, "Aligning supplied text", 0, audio.Duration()});
        const auto features = runtime.Features(source.samples);
        auto result = engine.Align(features, text, language);
        if (config.progress)
            config.progress(
                {din::common::ProgressStage::Aligning, "Alignment complete", audio.Duration(), audio.Duration()});
        return result;
    }
};
Qwen3ForcedAligner::Qwen3ForcedAligner(ForcedAlignerConfig config)
{
    impl_ = std::make_unique<Impl>(std::move(config));
}
Qwen3ForcedAligner::~Qwen3ForcedAligner() = default;
std::vector<WordTimestamp> Qwen3ForcedAligner::Align(const din::io::Audio& audio, const std::string& text,
                                                     const std::string& language)
{
    return impl_->AlignAudio(audio, text, language);
}
std::vector<WordTimestamp> Qwen3ForcedAligner::AlignFile(const std::filesystem::path& path, const std::string& text,
                                                         const std::string& language)
{
    if (impl_->config.progress)
        impl_->config.progress({din::common::ProgressStage::DecodingAudio, path.filename().string()});
    return Align(din::io::LoadAudio(path, kRate), text, language);
}
}  // namespace din::asr::qwen3
