// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/image.h"
#include "seedvr_pipeline.h"
#include <argparse/argparse.hpp>

namespace {
    din::seedvr::Config ParseArgs(int argc, char *argv[]) {
        din::seedvr::Config config;
        argparse::ArgumentParser parser("din_seedvr2");
        parser.add_description(
            "Upscale a PNG or a directory of PNGs with SeedVR2 using ONNX Runtime, TensorRT RTX, and CUDA.");
        parser.add_argument("input").help("Input PNG image or directory of PNG images.");
        parser.add_argument("--output")
                .default_value(config.output_path.string())
                .nargs(1)
                .metavar("PATH")
                .help("Output PNG for one input, or output directory for a directory input.");
        parser.add_argument("--model-dir").default_value(config.model_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--transformer-precision").default_value(config.transformer_precision)
                .nargs(1).metavar("fp16|fp8_wo|nvfp4")
                .help("DiT precision; VAE and conditioning remain FP16.");
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--scale").default_value(std::to_string(config.scale)).nargs(1).metavar("FLOAT");
        parser.add_argument("--color-correction")
                .default_value(config.color_correction).nargs(1).metavar("lab|none")
                .help("Postprocess against the resized input: lab (default) or none.");
        parser.add_argument("--seed").default_value(std::to_string(config.seed)).nargs(1).metavar("N");
        parser.add_argument("--rng")
                .default_value(std::string{"cpu-std"})
                .nargs(1)
                .metavar("cpu-std|cuda-philox|cpu-philox")
                .help(
                    "Noise generator: cpu-std (default), cuda-philox (CUDA only), or cpu-philox (fixed H100 SXM5 ordering).");
        parser.add_argument("--ws")
                .default_value(std::string{"off"})
                .nargs(1)
                .metavar("BUDGET")
                .help("DiT TensorRT RTX weight-streaming budget, forwarded to ORT; off by default.");
        parser.add_argument("--vae-encoder-tile-size").default_value(0).scan<'i', int>()
                .help("Encoder tile size in upscaled pixels, multiple of 16; 0 disables tiling.");
        parser.add_argument("--vae-encoder-overlap").default_value(64).scan<'i', int>()
                .help("Encoder overlap in upscaled pixels, multiple of 16.");
        parser.add_argument("--vae-decoder-tile-size").default_value(0).scan<'i', int>()
                .help("Decoder tile size in upscaled pixels, multiple of 16; 0 disables tiling.");
        parser.add_argument("--vae-decoder-overlap").default_value(64).scan<'i', int>()
                .help("Decoder overlap in upscaled pixels, multiple of 16.");
        try {
            parser.parse_args(argc, argv);
        } catch (const std::exception &exception) {
            std::cerr << parser << '\n';
            throw std::runtime_error(exception.what());
        }
        config.input_path = parser.get<std::string>("input");
        config.vae_encoder_tile_size = parser.get<int>("--vae-encoder-tile-size");
        config.vae_encoder_overlap = parser.get<int>("--vae-encoder-overlap");
        config.vae_decoder_tile_size = parser.get<int>("--vae-decoder-tile-size");
        config.vae_decoder_overlap = parser.get<int>("--vae-decoder-overlap");
        config.output_path = parser.get<std::string>("--output");
        config.model_dir = parser.get<std::string>("--model-dir");
        config.transformer_precision = parser.get<std::string>("--transformer-precision");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.scale = std::stof(parser.get<std::string>("--scale"));
        config.color_correction = parser.get<std::string>("--color-correction");
        if (config.color_correction != "lab" && config.color_correction != "none") {
            throw std::invalid_argument("--color-correction must be lab or none");
        }
        const std::string seed = parser.get<std::string>("--seed");
        size_t parsed = 0;
        const unsigned long parsed_seed = std::stoul(seed, &parsed, 10);
        if (parsed != seed.size()) {
            throw std::invalid_argument("--seed must be an unsigned integer");
        }
        config.seed = static_cast<uint32_t>(parsed_seed);
        const std::string rng = parser.get<std::string>("--rng");
        if (rng == "cpu-std")
            config.rng = din::common::Rng::CpuStd;
        else if (rng == "cuda-philox")
            config.rng = din::common::Rng::CudaPhilox;
        else if (rng == "cpu-philox")
            config.rng = din::common::Rng::CpuPhilox;
        else
            throw std::invalid_argument("--rng must be cpu-std, cuda-philox, or cpu-philox");
        config.weight_streaming_budget = parser.get<std::string>("--ws");
        if (config.weight_streaming_budget == "off")
            config.weight_streaming_budget.clear();
        return config;
    }

    bool IsPng(const std::filesystem::path &path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        return extension == ".png";
    }

    void RunOne(const din::seedvr::Config &config, din::seedvr::Pipeline &pipeline) {
        const auto start = std::chrono::steady_clock::now();
        din::seedvr::Image image = pipeline.Run(config.input_path);
        std::vector<float> bytes(image.chw.size());
        std::transform(image.chw.begin(), image.chw.end(), bytes.begin(),
                       [](float value) { return value * 255.0f; });
        if (!din::io::SaveRgbFloatImage(config.output_path.string(), bytes.data(), image.height, image.width,
                                        din::io::ImageValueRange::ZeroTo255)) {
            throw std::runtime_error("Failed to save SeedVR output: " + config.output_path.string());
        }
        const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
        std::cout << "Saved " << config.output_path.string() << " (" << image.width << 'x' << image.height << ") in "
                << duration.count() << " seconds" << std::endl;
    }
} // namespace

int main(int argc, char *argv[]) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try {
        din::seedvr::Config config = ParseArgs(argc, argv);
        std::cout << "RNG: " << (config.rng == din::common::Rng::CpuStd
                                     ? "cpu-std"
                                     : config.rng == din::common::Rng::CudaPhilox
                                           ? "cuda-philox"
                                           : "cpu-philox")
                << std::endl;
        if (std::filesystem::is_regular_file(config.input_path)) {
            if (!config.output_path.parent_path().empty()) {
                std::filesystem::create_directories(config.output_path.parent_path());
            }
            din::seedvr::Pipeline pipeline(config);
            RunOne(config, pipeline);
            return EXIT_SUCCESS;
        }
        if (!std::filesystem::is_directory(config.input_path)) {
            throw std::runtime_error("Input is not a PNG file or directory: " + config.input_path.string());
        }

        std::vector<std::filesystem::path> inputs;
        for (const auto &entry: std::filesystem::directory_iterator(config.input_path)) {
            if (entry.is_regular_file() && IsPng(entry.path())) {
                inputs.push_back(entry.path());
            }
        }
        std::sort(inputs.begin(), inputs.end());
        if (inputs.empty()) {
            throw std::runtime_error("No PNG images found in: " + config.input_path.string());
        }

        const std::filesystem::path default_output = din::seedvr::Config{}.output_path;
        if (config.output_path == default_output) {
            config.output_path = default_output.parent_path() / default_output.stem();
        }
        const std::filesystem::path output_dir = config.output_path;
        std::filesystem::create_directories(output_dir);
        if (std::filesystem::equivalent(config.input_path, output_dir)) {
            throw std::invalid_argument("The output directory must differ from the input directory");
        }

        din::seedvr::Pipeline pipeline(config);
        for (size_t index = 0; index < inputs.size(); ++index) {
            config.input_path = inputs[index];
            config.output_path = output_dir / (inputs[index].stem().string() + "_seedvr.png");
            std::cout << "[" << index + 1 << '/' << inputs.size() << "] " << inputs[index].string() << std::endl;
            RunOne(config, pipeline);
        }
        std::cout << "Completed " << inputs.size() << '/' << inputs.size() << " images" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception &exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
