// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "flux2_cli.h"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "flux2.h"
#include "io/image.h"
#include <argparse/argparse.hpp>

namespace
{
unsigned int parse_uint(const std::string& value, const char* flag_name)
{
    size_t parsed_chars = 0;
    const unsigned long parsed = std::stoul(value, &parsed_chars, 10);
    if (parsed_chars != value.size())
    {
        throw std::invalid_argument(std::string(flag_name) + " must be an unsigned integer");
    }
    return static_cast<unsigned int>(parsed);
}

std::string to_string(Flux2ProcessingBackend backend)
{
    switch (backend)
    {
    case Flux2ProcessingBackend::Cpu:
        return "cpu";
    case Flux2ProcessingBackend::Cuda:
        return "cuda";
    case Flux2ProcessingBackend::Dx:
        return "dx";
    case Flux2ProcessingBackend::DxCig:
        return "dx-cig";
    case Flux2ProcessingBackend::Vk:
        return "vk";
    case Flux2ProcessingBackend::VkCig:
        return "vk-cig";
    }
    return "unknown";
}

std::string to_string(Flux2ExecutionProvider provider)
{
    switch (provider)
    {
    case Flux2ExecutionProvider::Cpu:
        return "cpu";
    case Flux2ExecutionProvider::TrtRtx:
        return "trt-rtx";
    }
    return "unknown";
}

Flux2ProcessingBackend parse_processing_backend(std::string value)
{
    for (char& c : value)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "cpu")
    {
        return Flux2ProcessingBackend::Cpu;
    }
    if (value == "cuda")
    {
        return Flux2ProcessingBackend::Cuda;
    }
    if (value == "dx")
    {
        return Flux2ProcessingBackend::Dx;
    }
    if (value == "dx-cig")
    {
        return Flux2ProcessingBackend::DxCig;
    }
    if (value == "vk")
    {
        return Flux2ProcessingBackend::Vk;
    }
    if (value == "vk-cig")
    {
        return Flux2ProcessingBackend::VkCig;
    }
    throw std::invalid_argument("Processing must be one of: cpu, cuda, dx, dx-cig, vk, vk-cig");
}

Flux2ExecutionProvider parse_execution_provider(std::string value)
{
    for (char& c : value)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "cpu")
    {
        return Flux2ExecutionProvider::Cpu;
    }
    if (value == "trt-rtx" || value == "trt")
    {
        return Flux2ExecutionProvider::TrtRtx;
    }
    throw std::invalid_argument("Provider must be one of: cpu, trt-rtx");
}

std::string parse_precision(std::string value)
{
    for (char& c : value)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "bf16" || value == "fp8" || value == "nvfp4")
    {
        return value;
    }
    throw std::invalid_argument("Precision must be one of: bf16, fp8, nvfp4");
}

void validate_config(const Flux2Config& config)
{
    if (config.provider == Flux2ExecutionProvider::Cpu && config.processing != Flux2ProcessingBackend::Cpu)
    {
        throw std::invalid_argument("--provider cpu requires --processing cpu");
    }
}

Flux2Config parse_args(int argc, char* argv[])
{
    Flux2Config config;
    config.model_dir = DEFAULT_MODEL_BASE_PATH;
    config.prompt = DEFAULT_PROMPT;

    argparse::ArgumentParser parser("din_flux2");
    parser.add_description("Run Flux2 image generation with a selectable processing backend.");
    parser.add_argument("--processing")
        .default_value(to_string(config.processing))
        .nargs(1)
        .metavar("cpu|cuda|dx|dx-cig|vk|vk-cig")
        .help("Select the processing backend.");
    parser.add_argument("--provider")
        .default_value(to_string(config.provider))
        .nargs(1)
        .metavar("cpu|trt-rtx")
        .help("Select the ONNX Runtime execution provider.");
    parser.add_argument("--model-dir")
        .default_value(config.model_dir.string())
        .nargs(1)
        .metavar("PATH")
        .help("Root directory with shared Flux2 ONNX artifacts and transformer_<precision> directories.");
    parser.add_argument("--precision")
        .default_value(config.precision)
        .nargs(1)
        .metavar("bf16|fp8|nvfp4")
        .help("Transformer precision to load from transformer_<precision>.");
    parser.add_argument("--ep-cache")
          .default_value(config.ep_cache_dir.string())
          .nargs(1)
          .metavar("PATH")
          .help("TensorRT RTX runtime cache directory.");
    parser.add_argument("--ep-context-dir")
          .default_value(config.ep_context_dir.string())
          .nargs(1)
          .metavar("PATH")
          .help("Directory for ONNX Runtime EP-context models.");
    parser.add_argument("--output")
          .default_value(std::string{"."})
          .nargs(1)
          .metavar("DIR")
          .help("Directory where generated images are written.");
    parser.add_argument("--prompt")
          .default_value(config.prompt)
          .nargs(1)
          .metavar("TEXT")
          .help("Prompt text to encode with the Flux2 tokenizer.");
    parser.add_argument("--seed").default_value(std::to_string(config.seed)).nargs(1).metavar("N").help("Random seed.");
    parser.add_argument("--num-images")
          .default_value(std::to_string(config.num_images))
          .nargs(1)
          .metavar("N")
          .help("Number of images to generate.");

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception& exception)
    {
        std::cerr << parser << '\n';
        throw std::runtime_error(exception.what());
    }

    config.processing = parse_processing_backend(parser.get<std::string>("--processing"));
    config.provider = parse_execution_provider(parser.get<std::string>("--provider"));
    config.model_dir = parser.get<std::string>("--model-dir");
    config.precision = parse_precision(parser.get<std::string>("--precision"));
    config.ep_cache_dir = parser.get<std::string>("--ep-cache");
    config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
    config.prompt = parser.get<std::string>("--prompt");
    config.seed = parse_uint(parser.get<std::string>("--seed"), "--seed");
    config.num_images = parse_uint(parser.get<std::string>("--num-images"), "--num-images");

    const std::filesystem::path output_dir = parser.get<std::string>("--output");
    std::filesystem::create_directories(output_dir);
    config.output_path = output_dir / "flux2.png";
    validate_config(config);

    return config;
}

std::filesystem::path make_output_path(const Flux2Config& config, unsigned int image_index)
{
    const std::string stem = config.output_path.stem().string();
    const std::string extension = config.output_path.extension().string();
    return config.output_path.parent_path() / (stem + "_" + std::to_string(image_index) + extension);
}

void save_image(const std::filesystem::path& output_path, const Flux2Image& image)
{
    if (image.data.empty())
    {
        throw std::runtime_error("Backend returned an empty image");
    }
    if (!din::io::SaveRgbFloatImage(output_path.string(), image.data.data(), image.height, image.width,
                                    din::io::ImageValueRange::MinusOneToOne))
    {
        throw std::runtime_error("Failed to save image to " + output_path.string());
    }
    std::cout << "Image saved to " << output_path.string() << std::endl;
}
}  // namespace

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    try
    {
        const Flux2Config config = parse_args(argc, argv);
        std::cout << "Model dir: " << config.model_dir.string() << "\n"
            << "Precision: " << config.precision << "\n"
            << "Output: " << config.output_path.string() << "\n"
            << "Prompt: " << config.prompt << "\n"
            << "Seed:   " << config.seed << "\n"
            << "Num images: " << config.num_images << "\n"
            << "Processing: " << to_string(config.processing) << "\n"
            << "Provider: " << to_string(config.provider) << "\n"
            << "EP cache: " << config.ep_cache_dir.string() << "\n"
            << "EP context dir: " << config.ep_context_dir.string() << "\n"
                  << std::endl;

        auto pipeline = CreateFlux2Pipeline(config);
        pipeline->Initialize();

        for (unsigned int image_index = 0; image_index < config.num_images; ++image_index)
        {
            const unsigned int current_seed = config.seed + image_index;
            std::cout << "\n========== Image " << (image_index + 1) << "/" << config.num_images
                      << " (seed=" << current_seed << ") ==========" << std::endl;
            const auto image_start = std::chrono::steady_clock::now();
            Flux2Image image = pipeline->GenerateImage(current_seed);
            const auto image_end = std::chrono::steady_clock::now();
            save_image(make_output_path(config, image_index), image);
            const std::chrono::duration<double> image_duration = image_end - image_start;
            std::cout << "Image " << (image_index + 1) << " completed in " << image_duration.count() << " seconds"
                      << std::endl;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return EXIT_FAILURE;
}
