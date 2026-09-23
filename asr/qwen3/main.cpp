// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>

#include "qwen3.h"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
    using namespace din::asr::qwen3;
    try
    {
        Qwen3Config config;
        argparse::ArgumentParser parser("din_asr_qwen3_cli");
        parser.add_description("Offline Qwen3 ASR.");
        parser.add_argument("audiofile");
        parser.add_argument("--provider").default_value(config.provider).choices("cpu", "trt-rtx");
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--lang-id")
            .default_value(config.lang_id)
            .help("Language code or name (case-insensitive); auto detects ASR language");
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string());
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string());
        parser.add_argument("--max-new-tokens").default_value(config.max_new_tokens).scan<'i', int>();
        parser.add_argument("--max-chunk-seconds")
            .default_value(config.max_chunk_seconds)
            .scan<'i', int>()
            .help("Chunk target: 0 = auto (1200 s), limited by KV capacity; boundaries may add 5 s");
        parser.parse_args(argc, argv);
        config.provider = parser.get<std::string>("--provider");
        config.model_dir = parser.get<std::string>("--model-dir");
        config.lang_id = parser.get<std::string>("--lang-id");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.max_new_tokens = parser.get<int>("--max-new-tokens");
        config.max_chunk_seconds = parser.get<int>("--max-chunk-seconds");
        Qwen3Pipeline pipeline(std::move(config));
        const auto result = pipeline.TranscribeFile(parser.get<std::string>("audiofile"));
        nlohmann::json output{
            {"transcription", result.text},          {"language", result.language},
            {"reached_eos", result.reached_eos},     {"tokens", result.tokens},
            {"audio_seconds", result.audio_seconds}, {"transcribe_seconds", result.transcribe_seconds}};
        output["segments"] = nlohmann::json::array();
        output["sample_rate"] = 16000;
        output["chunks_processed"] = result.chunks_processed;
        for (const auto& segment : result.segments)
            output["segments"].push_back({{"text", segment.text},
                                          {"language", segment.language},
                                          {"start_sample", segment.start_sample},
                                          {"end_sample", segment.end_sample}});
        std::cout << output.dump(2) << '\n';
        return result.reached_eos ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Qwen3: " << error.what() << '\n';
        return 1;
    }
}
