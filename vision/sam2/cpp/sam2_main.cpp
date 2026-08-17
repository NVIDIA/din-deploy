// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "argparse/argparse.hpp"
#include "sam2_io.h"
#include "sam2_pipeline.h"
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;

    try
    {
        din::sam2::Sam2PipelineConfig config;

        argparse::ArgumentParser parser("din_sam2");
        parser.add_description("Run exported SAM2 ONNX graphs on an image or frame sequence.");
        parser.add_argument("input").help("Input PNG file or directory of frame images.");
        parser.add_argument("prompt").help("Prompt JSON file.");
        parser.add_argument("output").help("Output directory for mask PNGs.");
        parser.add_argument("--model-dir")
            .default_value(config.model_dir.string())
            .metavar("PATH")
            .help("Directory with exported SAM2 ONNX artifacts.");
        parser.add_argument("--ep-cache")
            .default_value(config.ep_cache_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX runtime cache directory.");
        parser.add_argument("--ep-context-dir")
            .default_value(config.ep_context_dir.string())
            .metavar("PATH")
            .help("TensorRT RTX embedded context model directory.");
        parser.add_argument("--provider")
            .default_value(config.provider)
            .metavar("cpu|trt-rtx")
            .help("Execution provider.");
        parser.add_argument("--propagate")
            .default_value(false)
            .implicit_value(true)
            .help("Enable video propagation with memory attention/encoder graphs.");
        parser.add_argument("--max-frames")
            .default_value(static_cast<size_t>(0))
            .scan<'u', size_t>()
            .metavar("N")
            .help("Limit the number of input frames (0 = all).");
        parser.add_argument("--dump-masks")
            .default_value(false)
            .implicit_value(true)
            .help("Write mask PNGs to the output directory.");

        try
        {
            parser.parse_args(argc, argv);
        }
        catch (const std::exception& exception)
        {
            std::cerr << parser << '\n';
            throw std::runtime_error(exception.what());
        }

        const fs::path input_path = parser.get<std::string>("input");
        const fs::path prompt_path = parser.get<std::string>("prompt");
        const fs::path output_path = parser.get<std::string>("output");
        config.model_dir = fs::path{parser.get<std::string>("--model-dir")};
        config.ep_cache_dir = fs::path{parser.get<std::string>("--ep-cache")};
        config.ep_context_dir = fs::path{parser.get<std::string>("--ep-context-dir")};
        config.provider = parser.get<std::string>("--provider");
        config.propagate = parser.get<bool>("--propagate");
        config.dump_masks = parser.get<bool>("--dump-masks");
        const size_t max_frames = parser.get<size_t>("--max-frames");

        const auto frames = din::sam2::ResolveInputFrames(input_path, max_frames);
        const auto first = din::sam2::LoadPng(frames.front());
        const auto metadata = din::sam2::LoadSam2Config(config.model_dir / "metadata.json");
        auto prompt = din::sam2::LoadPrompt(prompt_path, first.height, first.width, metadata.image_size);

        const auto session_load_start = std::chrono::steady_clock::now();
        din::sam2::Sam2Pipeline pipeline(std::move(config));
        const auto session_load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - session_load_start).count();

        const auto processing_start = std::chrono::steady_clock::now();
        pipeline.RunBatch(frames, prompt, output_path);
        const auto processing_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - processing_start).count();
        const double fps = processing_seconds > 0.0 ? static_cast<double>(frames.size()) / processing_seconds : 0.0;

        std::cout << std::fixed << std::setprecision(3) << "Session load: " << session_load_seconds << " s\n"
                  << "Processing: " << processing_seconds << " s (" << fps << " FPS)\n";
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "ONNX Runtime error: " << exception.what() << '\n';
        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
