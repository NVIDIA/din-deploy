// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "argparse/argparse.hpp"
#include "whisper.h"

template <class Pipeline, class Config>
int RunCli(int argc, char** argv, const din::asr::whisper::CliSpec& spec)
{
    namespace fs = std::filesystem;
    try
    {
        const auto total_start = std::chrono::steady_clock::now();

        Config config;
        din::asr::whisper::TranscriptionOptions options;

        argparse::ArgumentParser parser(spec.fallback_name);
        parser.add_description("Transcribe audio with exported " + spec.artifact_label + " ONNX artifacts.");
        parser.add_argument("audiofile").help("Audio file to transcribe.");
        parser.add_argument("--model-dir")
            .default_value(config.model_dir.string())
            .metavar("PATH")
            .help("Directory with exported " + spec.artifact_label + " ONNX artifacts.");
        parser.add_argument("--provider")
            .default_value(spec.default_provider)
            .metavar("cpu|trt-rtx")
            .help("Execution provider.");
        parser.add_argument("--timestamps", "--timesteps")
            .default_value(options.timestamps)
            .metavar(spec.timestamps_modes)
            .help("Timestamp output mode.");
        parser.add_argument("--ep-cache")
            .default_value(config.ep_cache_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX runtime cache directory.");
        parser.add_argument("--ep-context-dir")
            .default_value(config.ep_context_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX embedded context model directory.");
        if constexpr (requires(Config value) { value.encoder_profile_frames = int64_t{}; })
        {
            parser.add_argument("--encoder-profile-frames")
                .default_value(config.encoder_profile_frames)
                .template scan<'i', int64_t>()
                .metavar("VALUE")
                .help("TensorRT RTX max encoder profile frames.");
        }
        if constexpr (requires(Config value) { value.lang_id = std::string{}; })
        {
            auto lang_modes = std::string{"LANG"};
            auto lang_help = std::string{"Nemotron target language/prompt id."};
            if (!spec.allowed_lang_ids.empty())
            {
                lang_modes.clear();
                for (const auto& lang_id : spec.allowed_lang_ids)
                {
                    if (!lang_modes.empty())
                    {
                        lang_modes += "|";
                    }
                    lang_modes += lang_id;
                }
                lang_help += " Choices: " + lang_modes + ".";
            }
            parser.add_argument("--lang-id", "--lang_id")
                .default_value(config.lang_id)
                .metavar(lang_modes)
                .help(lang_help);
        }
        if constexpr (requires(Config value) { value.disable_cuda_sampling = bool{}; })
        {
            parser.add_argument("--cpu-sampling", "--disable-cuda-sampling")
                .default_value(false)
                .implicit_value(true)
                .help("Force greedy argmax on the CPU (disable the CUDA sampling kernel).");
        }

        try
        {
            parser.parse_args(argc, argv);
        }
        catch (const std::exception& exception)
        {
            std::cerr << parser << '\n';
            throw std::runtime_error(exception.what());
        }

        const fs::path audio_file = parser.get<std::string>("audiofile");
        config.model_dir = fs::path{parser.get<std::string>("--model-dir")};
        config.provider = parser.get<std::string>("--provider");
        options.timestamps = parser.get<std::string>("--timestamps");
        if (!spec.allowed_timestamps.empty() &&
            std::find(spec.allowed_timestamps.begin(), spec.allowed_timestamps.end(), options.timestamps) ==
                spec.allowed_timestamps.end())
        {
            throw std::runtime_error("unsupported timestamp mode: " + options.timestamps);
        }
        config.ep_cache_dir = fs::path{parser.get<std::string>("--ep-cache")};
        config.ep_context_dir = fs::path{parser.get<std::string>("--ep-context-dir")};
        if constexpr (requires(Config value) { value.encoder_profile_frames = int64_t{}; })
        {
            config.encoder_profile_frames = parser.get<int64_t>("--encoder-profile-frames");
        }
        if constexpr (requires(Config value) { value.lang_id = std::string{}; })
        {
            config.lang_id = parser.get<std::string>("--lang-id");
            if (!spec.allowed_lang_ids.empty() && std::find(spec.allowed_lang_ids.begin(), spec.allowed_lang_ids.end(),
                                                            config.lang_id) == spec.allowed_lang_ids.end())
            {
                throw std::runtime_error("unsupported lang-id: " + config.lang_id);
            }
        }
        if constexpr (requires(Config value) { value.disable_cuda_sampling = bool{}; })
        {
            config.disable_cuda_sampling = parser.get<bool>("--cpu-sampling");
        }

        const auto provider = config.provider;
        const auto load_start = std::chrono::steady_clock::now();
        Pipeline pipeline(std::move(config));
        const auto load_seconds = SecondsSince(load_start);

        const auto result = pipeline.TranscribeFile(audio_file);
        pipeline.Print(std::cout, result, options);

        std::cout << "\nprovider: " << provider << '\n';
        std::cout << "audio: " << result.audio_seconds << "s\n";
        std::cout << "load: " << load_seconds << "s\n";
        std::cout << "transcribe: " << result.transcribe_seconds << "s\n";
        std::cout << "encode: " << result.encode_seconds << "s\n";
        std::cout << "greedy: " << result.greedy_seconds << "s\n";
        std::cout << "speed: " << (result.audio_seconds / result.transcribe_seconds) << "x\n";
        std::cout << "total: " << SecondsSince(total_start) << "s\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    const din::asr::whisper::CliSpec spec{
        .fallback_name = "din_asr_whisper",
        .artifact_label = "Whisper",
        .default_provider = "trt-rtx",
        .timestamps_modes = "none",
        .allowed_timestamps = {"none"},
    };
    return RunCli<din::asr::whisper::WhisperPipeline, din::asr::whisper::WhisperConfig>(argc, argv, spec);
}
