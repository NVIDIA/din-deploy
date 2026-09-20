// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "argparse/argparse.hpp"
#include "whisper.h"

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;
    using namespace din::asr::whisper;
    try
    {
        const auto total_start = std::chrono::steady_clock::now();

        WhisperConfig config;
        din::asr::whisper::TranscriptionOptions options;

        argparse::ArgumentParser parser("din_asr_whisper");
        parser.add_argument("--no-context")
            .default_value(false)
            .implicit_value(true)
            .help("Do not condition long-form windows on previous text.");
        parser.add_description("Transcribe audio with exported Whisper ONNX artifacts.");
        parser.add_argument("--prefill-block-size")
            .default_value(128)
            .scan<'i', int>()
            .help("History bucket size for the shared decoder (default 128; 0 disables buckets).");
        parser.add_argument("--repeat")
            .default_value(1)
            .scan<'i', int>()
            .help("Transcribe the audio N times using the same pipeline; report each run separately.");
        parser.add_argument("audiofile").help("Audio file to transcribe.");
        parser.add_argument("--model-dir")
            .default_value(config.model_dir.string())
            .metavar("PATH")
            .help("Directory with exported Whisper ONNX artifacts.");
        parser.add_argument("--provider")
            .default_value(config.provider)
            .metavar("cpu|trt-rtx")
            .help("Execution provider.");
        parser.add_argument("--timestamps", "--timesteps")
            .default_value(options.timestamps)
            .metavar("none|segment|json")
            .help("Timestamp output mode.");
        parser.add_argument("--ep-cache")
            .default_value(config.ep_cache_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX runtime cache directory.");
        parser.add_argument("--ep-context-dir")
            .default_value(config.ep_context_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX embedded context model directory.");
        parser.add_argument("--lang-id", "--lang_id")
            .default_value(config.lang_id)
            .metavar("LANG|auto")
            .help("Whisper language code, or auto to detect once per recording.");
        parser.add_argument("--cpu-sampling", "--disable-cuda-sampling")
            .default_value(false)
            .implicit_value(true)
            .help("Force greedy argmax on the CPU (disable the CUDA sampling kernel).");

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
        if (options.timestamps != "none" && options.timestamps != "segment" && options.timestamps != "json")
        {
            throw std::runtime_error("unsupported timestamp mode: " + options.timestamps);
        }
        config.ep_cache_dir = fs::path{parser.get<std::string>("--ep-cache")};
        config.ep_context_dir = fs::path{parser.get<std::string>("--ep-context-dir")};
        config.lang_id = parser.get<std::string>("--lang-id");
        config.disable_cuda_sampling = parser.get<bool>("--cpu-sampling");
        config.condition_on_previous_text = !parser.get<bool>("--no-context");
        config.prefill_block_size = parser.get<int>("--prefill-block-size");

        const int repeat = parser.get<int>("--repeat");
        if (repeat < 1)
            throw std::runtime_error("--repeat must be at least 1");

        const auto provider = config.provider;
        const auto load_start = std::chrono::steady_clock::now();
        WhisperPipeline pipeline(std::move(config));
        const auto load_seconds = SecondsSince(load_start);

        std::cout << "load: " << load_seconds << "s\n";
        for (int run = 0; run < repeat; ++run)
        {
            if (repeat > 1)
                std::cout << "\nrun: " << run + 1 << '/' << repeat << '\n';
            const auto result = pipeline.TranscribeFile(audio_file);
            pipeline.Print(std::cout, result, options);

            std::cout << "\nprovider: " << provider << '\n';
            std::cout << "audio: " << result.audio_seconds << "s\n";
            std::cout << "transcribe: " << result.transcribe_seconds << "s\n";
            std::cout << "encode: " << result.encode_seconds << "s\n";
            std::cout << "greedy: " << result.greedy_seconds << "s\n";
            if (result.model_window_seconds > 0.0)
            {
                std::cout << "model window audio: " << result.model_window_seconds << "s\n";
            }
            std::cout << "speed: " << (result.audio_seconds / result.transcribe_seconds) << "x\n";
        }
        std::cout << "total: " << SecondsSince(total_start) << "s\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
