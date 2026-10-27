// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <future>
#include <iostream>

#include "argparse/argparse.hpp"
#include "diarization.h"
#include "forced_aligner.h"
#include "qwen3.h"
#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
    using namespace din::asr;
    try
    {
        diarization::Config config;
        argparse::ArgumentParser parser("din_diarization_cli");
        parser.add_argument("audiofile");
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--provider").default_value(config.provider).choices("cpu", "trt-rtx");
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string());
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string());
        parser.add_argument("--threshold").default_value(config.threshold).scan<'g', float>();
        parser.add_argument("--asr-dir").default_value(std::string{}).help("Optional Qwen3 ASR export");
        parser.add_argument("--aligner-dir").default_value(std::string{}).help("Optional Qwen3 forced aligner export");
        parser.add_argument("--asr-provider").default_value(std::string{"trt-rtx"}).choices("cpu", "trt-rtx");
        parser.add_argument("--aligner-provider").default_value(std::string{}).choices("", "cpu", "trt-rtx");
        parser.add_argument("--lang-id").default_value(std::string{"auto"});
        parser.add_argument("--repeat").default_value(1).scan<'i', int>().help("Reuse loaded engines for timing");
        parser.parse_args(argc, argv);
        config.model_dir = parser.get<std::string>("--model-dir");
        config.provider = parser.get<std::string>("--provider");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.threshold = parser.get<float>("--threshold");
        const auto audio = din::io::LoadAudio(parser.get<std::string>("audiofile"), 16000);
        const auto asr_dir = parser.get<std::string>("--asr-dir");
        const auto aligner_dir = parser.get<std::string>("--aligner-dir");
        if (!aligner_dir.empty() && asr_dir.empty())
            throw std::invalid_argument("--aligner-dir requires --asr-dir");
        const int repeat = parser.get<int>("--repeat");
        if (repeat < 1)
            throw std::invalid_argument("--repeat must be positive");
        diarization::Pipeline diarizer(config);
        std::unique_ptr<qwen3::Qwen3Pipeline> asr;
        std::unique_ptr<qwen3::Qwen3ForcedAligner> aligner;
        if (!asr_dir.empty())
        {
            qwen3::Qwen3Config c;
            c.model_dir = asr_dir;
            c.max_chunk_seconds = aligner_dir.empty() ? 0 : 175;
            c.provider = parser.get<std::string>("--asr-provider");
            c.lang_id = parser.get<std::string>("--lang-id");
            asr = std::make_unique<qwen3::Qwen3Pipeline>(c);
            if (!aligner_dir.empty())
            {
                qwen3::ForcedAlignerConfig alignment;
                alignment.model_dir = aligner_dir;
                alignment.provider = parser.get<std::string>("--aligner-provider");
                if (alignment.provider.empty())
                    alignment.provider = c.provider;
                aligner = std::make_unique<qwen3::Qwen3ForcedAligner>(alignment);
            }
        }
        nlohmann::json output;
        for (int iteration = 0; iteration < repeat; ++iteration)
        {
            const auto start = std::chrono::steady_clock::now();
            auto task = std::async(std::launch::async,
                                   [&]
                                   {
                                       return diarizer.Diarize(audio);
                                   });
            qwen3::TranscriptionResult transcript;
            if (asr)
                transcript = asr->Transcribe(audio);
            std::vector<qwen3::WordTimestamp> timestamps;
            if (aligner)
            {
                std::vector<qwen3::AlignmentSegment> segments;
                for (const auto& segment : transcript.segments)
                    segments.push_back({segment.text, segment.start_sample, segment.end_sample, segment.language});
                timestamps = aligner->AlignSegments(audio, segments);
            }
            const auto result = task.get();
            const float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
            output = {{"audio_seconds", result.audio_seconds},
                      {"diarize_seconds", result.process_seconds},
                      {"process_seconds", elapsed},
                      {"rtf", elapsed / result.audio_seconds},
                      {"segments", nlohmann::json::array()}};
            for (const auto& segment : result.segments)
                output["segments"].push_back(
                    {{"start_time", segment.start_time}, {"end_time", segment.end_time}, {"speaker", segment.speaker}});
            if (asr)
            {
                output["transcription"] = transcript.text;
                output["reached_eos"] = transcript.reached_eos;
                output["timestamps"] = nlohmann::json::array();
                for (const auto& word : timestamps)
                    output["timestamps"].push_back({{"text", word.text},
                                                    {"start_time", word.start_time},
                                                    {"end_time", word.end_time},
                                                    {"speaker", result.SpeakerAt(word.start_time, word.end_time)}});
            }
            std::cerr << "Run " << iteration + 1 << ": " << elapsed << " s, RTF " << elapsed / result.audio_seconds
                      << '\n';
        }
        std::cout << output.dump(2) << '\n';
        return asr && !output.at("reached_eos").get<bool>() ? 1 : 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Diarization: " << error.what() << '\n';
        return 1;
    }
}
