// SPDX-License-Identifier: Apache-2.0
#include "diarization.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "nvtx_helper.h"
#include "ort_session.h"
#include <nlohmann/json.hpp>

namespace din::asr::diarization
{
using din::common::OrtRunner;
using din::common::TensorBuffer;

struct Pipeline::Impl
{
    static constexpr int64_t kSamples = 3039 * 160 + 512 + 1;
    Config config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "diarization"};
    std::unique_ptr<OrtRunner> model;
    std::unique_ptr<TensorBuffer<float>> history, history_preds, next_history, next_history_preds, probabilities,
        samples;
    std::unique_ptr<TensorBuffer<int64_t>> feature_length, history_length, sample_bounds;
    Ort::IoBinding binding{nullptr};
    Ort::RunOptions options;

    explicit Impl(Config c)
        : config(std::move(c))
    {
        if (!std::isfinite(config.threshold) || config.threshold <= 0 || config.threshold >= 1)
            throw std::invalid_argument("Diarization threshold must be between 0 and 1");
        std::ifstream file(config.model_dir / "metadata.json");
        if (!file)
            throw std::runtime_error("Cannot read diarization metadata");
        const auto metadata = nlohmann::json::parse(file);
        if (metadata.at("format_version") != 3 || metadata.at("chunk_frames") != 340 ||
            metadata.at("cache_frames") != 264 || metadata.at("speakers") != 8)
            throw std::runtime_error("Incompatible diarization export; re-export the model");
        if (config.provider == "cpu" && metadata.at("dtype") != "fp32")
            throw std::runtime_error("Use an FP32 diarization export with the CPU provider");
        const din::common::EpContextOptions context{config.ep_context_dir.string()};
        if (config.provider == "trt-rtx")
            din::common::RegisterTensorRTRTXProvider(env);
        din::common::ModelProfile profile;
        profile.cache_subpath = "diarization_" + metadata.at("graph_hash").get<std::string>();
        model = std::make_unique<OrtRunner>(env, (config.model_dir / "diarization.onnx").string(), config.provider,
                                            config.ep_cache_dir.string(), context, profile);
        const bool device = model->HasDeviceIo();
        history = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, 304, 512}, device);
        next_history = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, 304, 512}, device);
        history_preds = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, 264, 8}, device);
        next_history_preds = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, 264, 8}, device);
        probabilities = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, 2720, 8}, device);
        feature_length = std::make_unique<TensorBuffer<int64_t>>(*model, std::vector<int64_t>{1}, device);
        history_length = std::make_unique<TensorBuffer<int64_t>>(*model, std::vector<int64_t>{1}, device);
        samples = std::make_unique<TensorBuffer<float>>(*model, std::vector<int64_t>{1, kSamples}, device);
        sample_bounds = std::make_unique<TensorBuffer<int64_t>>(*model, std::vector<int64_t>{2}, device);
        binding = Ort::IoBinding(model->session);
        binding.BindInput("samples", samples->BindingValue());
        binding.BindInput("sample_bounds", sample_bounds->BindingValue());
        binding.BindInput("feature_length", feature_length->BindingValue());
        binding.BindInput("history", history->BindingValue());
        binding.BindInput("history_preds", history_preds->BindingValue());
        binding.BindInput("history_length", history_length->BindingValue());
        binding.BindOutput("probabilities", probabilities->BindingValue());
        binding.BindOutput("next_history", next_history->BindingValue());
        binding.BindOutput("next_history_preds", next_history_preds->BindingValue());
        if (device)
            options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    }

    Result Run(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"diarization"};
        if (audio.sample_rate != 16000 || audio.samples.empty())
            throw std::invalid_argument("Expected nonempty 16 kHz mono audio");
        const auto start = std::chrono::steady_clock::now();
        Result result;
        result.audio_seconds = static_cast<float>(audio.Duration());
        const auto frames = audio.samples.size() / 160;
        if (!frames)
            return result;
        history->Fill(0);
        history_preds->Fill(0);
        history->CopyAsyncToDevice();
        history_preds->CopyAsyncToDevice();
        std::array<float, 8> active;
        active.fill(-1);
        for (size_t offset = 0; offset < frames; offset += 2720)
        {
            din::common::nvtx_scoped_range chunk_range{"diarization.chunk"};
            const auto count = std::min(size_t{3040}, frames - offset);
            const int64_t begin = static_cast<int64_t>(offset) * 160 - 257;
            const int64_t lo = std::max(int64_t{0}, begin);
            const int64_t hi = std::min(static_cast<int64_t>(audio.samples.size()), begin + kSamples);
            samples->Fill(0);
            std::copy_n(audio.samples.data() + lo, hi - lo, samples->HostData() + lo - begin);
            sample_bounds->HostData()[0] = lo - begin;
            sample_bounds->HostData()[1] = hi - begin;
            *feature_length->HostData() = static_cast<int64_t>(count);
            *history_length->HostData() = offset ? 304 : 0;
            samples->CopyAsyncToDevice();
            sample_bounds->CopyAsyncToDevice();
            feature_length->CopyAsyncToDevice();
            history_length->CopyAsyncToDevice();
            model->session.Run(options, binding);
            history->CopyFrom(*next_history);
            history_preds->CopyFrom(*next_history_preds);
            probabilities->CopyAsyncToHostWithNotification().Sync();
            const auto count_output = std::min(size_t{2720}, frames - offset);
            for (size_t frame = 0; frame < count_output; ++frame)
            {
                const float time = std::min(result.audio_seconds, static_cast<float>(offset + frame) * 0.01f);
                for (int speaker = 0; speaker < 8; ++speaker)
                {
                    const float probability = probabilities->HostData()[frame * 8 + speaker];
                    if (probability > config.threshold && active[speaker] < 0)
                        active[speaker] = time;
                    else if (probability < config.threshold && active[speaker] >= 0)
                    {
                        if (time > active[speaker])
                            result.segments.push_back({active[speaker], time, speaker});
                        active[speaker] = -1;
                    }
                }
            }
        }
        for (int speaker = 0; speaker < 8; ++speaker)
            if (active[speaker] >= 0 && active[speaker] < result.audio_seconds)
                result.segments.push_back({active[speaker], static_cast<float>(frames) * 0.01f, speaker});
        std::sort(result.segments.begin(), result.segments.end(),
                  [](const auto& a, const auto& b)
                  {
                      return a.start_time < b.start_time || (a.start_time == b.start_time && a.speaker < b.speaker);
                  });
        result.process_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        return result;
    }
};

int Result::SpeakerAt(float start, float end) const
{
    if (!std::isfinite(start) || !std::isfinite(end) || end < start)
        return -1;
    // Aligners can emit zero-duration words; use the speaker active at that instant.
    if (start == end)
        end = std::nextafter(end, std::numeric_limits<float>::infinity());
    std::array<float, 8> overlap{};
    for (const auto& segment : segments)
    {
        if (segment.start_time >= end)
            break;
        overlap[segment.speaker] +=
            std::max(0.0f, std::min(end, segment.end_time) - std::max(start, segment.start_time));
    }
    const auto best = std::max_element(overlap.begin(), overlap.end());
    return *best > 0 ? static_cast<int>(best - overlap.begin()) : -1;
}

Pipeline::Pipeline(Config config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}
Pipeline::~Pipeline() = default;
Result Pipeline::Diarize(const din::io::Audio& audio)
{
    return impl_->Run(audio);
}
Result Pipeline::DiarizeFile(const std::filesystem::path& path)
{
    return Diarize(din::io::LoadAudio(path, 16000));
}
}  // namespace din::asr::diarization
