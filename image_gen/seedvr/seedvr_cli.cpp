// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "io/image.h"
#include "seedvr_pipeline.h"
#include <argparse/argparse.hpp>

namespace
{
    din::seedvr::Config ParseArgs(int argc, char* argv[])
    {
        din::seedvr::Config config;
        argparse::ArgumentParser parser("din_seedvr2");
        parser.add_description("Upscale one PNG with SeedVR2 using ONNX Runtime, TensorRT RTX, and CUDA.");
        parser.add_argument("input").help("Input PNG image.");
        parser.add_argument("--output").default_value(config.output_path.string()).nargs(1).metavar("PNG");
        parser.add_argument("--model-dir").default_value(config.model_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string()).nargs(1).metavar("DIR");
        parser.add_argument("--scale").default_value(std::to_string(config.scale)).nargs(1).metavar("FLOAT");
        parser.add_argument("--seed").default_value(std::to_string(config.seed)).nargs(1).metavar("N");
        try
        {
            parser.parse_args(argc, argv);
        }
        catch (const std::exception& exception)
        {
            std::cerr << parser << '\n';
            throw std::runtime_error(exception.what());
        }
        config.input_path = parser.get<std::string>("input");
        config.output_path = parser.get<std::string>("--output");
        config.model_dir = parser.get<std::string>("--model-dir");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.scale = std::stof(parser.get<std::string>("--scale"));
        const std::string seed = parser.get<std::string>("--seed");
        size_t parsed = 0;
        const unsigned long parsed_seed = std::stoul(seed, &parsed, 10);
        if (parsed != seed.size())
        {
            throw std::invalid_argument("--seed must be an unsigned integer");
        }
        config.seed = static_cast<uint32_t>(parsed_seed);
        return config;
    }
} // namespace

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const din::seedvr::Config config = ParseArgs(argc, argv);
        if (!config.output_path.parent_path().empty())
        {
            std::filesystem::create_directories(config.output_path.parent_path());
        }
        const auto start = std::chrono::steady_clock::now();
        din::seedvr::Pipeline pipeline(config);
        din::seedvr::Image image = pipeline.Run();
        std::vector<float> bytes(image.chw.size());
        std::transform(image.chw.begin(), image.chw.end(), bytes.begin(),
                       [](float value)
                       {
                           return value * 255.0f;
                       });
        if (!din::io::SaveRgbFloatImage(config.output_path.string(), bytes.data(), image.height, image.width,
                                        din::io::ImageValueRange::ZeroTo255))
        {
            throw std::runtime_error("Failed to save SeedVR output: " + config.output_path.string());
        }
        const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
        std::cout << "Saved " << config.output_path.string() << " (" << image.width << 'x' << image.height << ") in "
            << duration.count() << " seconds" << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
