// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pixeldit_cli.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "io/image.h"
#include "pixeldit.h"
#include <argparse/argparse.hpp>

namespace din::image_gen
{
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

float parse_float(const std::string& value, const char* flag_name)
{
    size_t parsed_chars = 0;
    const float parsed = std::stof(value, &parsed_chars);
    if (parsed_chars != value.size())
    {
        throw std::invalid_argument(std::string(flag_name) + " must be a number");
    }
    return parsed;
}

std::string to_lower(std::string value)
{
    for (char& c : value)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string to_string(PixelDiTProcessingBackend backend)
{
    switch (backend)
    {
    case PixelDiTProcessingBackend::Cpu:
        return "cpu";
    case PixelDiTProcessingBackend::Cuda:
        return "cuda";
    }
    return "unknown";
}

std::string to_string(PixelDiTExecutionProvider provider)
{
    switch (provider)
    {
    case PixelDiTExecutionProvider::TrtRtx:
        return "trt-rtx";
    }
    return "unknown";
}

PixelDiTProcessingBackend parse_processing_backend(const std::string& value)
{
    const std::string lower = to_lower(value);
    if (lower == "cpu")
    {
        return PixelDiTProcessingBackend::Cpu;
    }
    if (lower == "cuda")
    {
        return PixelDiTProcessingBackend::Cuda;
    }
    if (lower == "dx" || lower == "dx-cig" || lower == "vk")
    {
        throw std::invalid_argument("--processing " + lower + " is not implemented for PixelDiT yet; use cpu or cuda");
    }
    throw std::invalid_argument("Processing must be one of: cpu, cuda");
}

PixelDiTExecutionProvider parse_execution_provider(const std::string& value)
{
    const std::string lower = to_lower(value);
    if (lower == "trt-rtx" || lower == "trt")
    {
        return PixelDiTExecutionProvider::TrtRtx;
    }
    if (lower == "cpu")
    {
        throw std::invalid_argument("--provider cpu is not supported for PixelDiT; use trt-rtx");
    }
    throw std::invalid_argument("Provider must be: trt-rtx");
}

PixelDiTConfig parse_args(int argc, char* argv[])
{
    PixelDiTConfig config;
    config.model_dir = DEFAULT_PIXELDIT_MODEL_DIR;
    config.prompt = DEFAULT_PIXELDIT_PROMPT;

    argparse::ArgumentParser parser("din_pixeldit");
    parser.add_description("Run PixelDiT text-to-image generation with a selectable processing backend.");
    parser.add_argument("--processing")
        .default_value(to_string(config.processing))
        .nargs(1)
        .metavar("cpu|cuda")
        .help("Select the processing backend.");
    parser.add_argument("--provider")
        .default_value(to_string(config.provider))
        .nargs(1)
        .metavar("trt-rtx")
        .help("Select the ONNX Runtime execution provider.");
    parser.add_argument("--model-dir")
        .default_value(config.model_dir.string())
        .nargs(1)
        .metavar("PATH")
        .help("Directory with exported PixelDiT ONNX artifacts (text_encoder, transformer, tokenizer, "
              "pipeline_config.json).");
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
    parser.add_argument("--prompt").default_value(config.prompt).nargs(1).metavar("TEXT").help("Prompt text.");
    parser.add_argument("--negative-prompt")
        .nargs(1)
        .metavar("TEXT")
        .help("Negative prompt for classifier-free guidance (default: from pipeline_config.json).");
    parser.add_argument("--seed").default_value(std::to_string(config.seed)).nargs(1).metavar("N").help("Random seed.");
    parser.add_argument("--num-images")
        .default_value(std::to_string(config.num_images))
        .nargs(1)
        .metavar("N")
        .help("Number of images to generate (seeds seed, seed+1, ...).");
    parser.add_argument("--steps").nargs(1).metavar("N").help("Sampling steps (default: from pipeline_config.json).");
    parser.add_argument("--cfg-scale").nargs(1).metavar("X").help("CFG scale (default: from pipeline_config.json).");
    parser.add_argument("--flow-shift").nargs(1).metavar("X").help("Flow shift (default: from pipeline_config.json).");
    parser.add_argument("--init-noise")
        .nargs(1)
        .metavar("FILE")
        .help("Validation: raw fp32 [1, 3, H, W] initial noise used instead of the seeded generator.");
    parser.add_argument("--dump-dir")
        .nargs(1)
        .metavar("DIR")
        .help("Validation: write raw tokens, prompt embeddings, timesteps, noise and final sample here.");

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
    config.ep_cache_dir = parser.get<std::string>("--ep-cache");
    config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
    config.prompt = parser.get<std::string>("--prompt");
    config.seed = parse_uint(parser.get<std::string>("--seed"), "--seed");
    config.num_images = parse_uint(parser.get<std::string>("--num-images"), "--num-images");
    if (auto value = parser.present<std::string>("--negative-prompt"))
    {
        config.negative_prompt = *value;
    }
    if (auto value = parser.present<std::string>("--steps"))
    {
        config.steps = static_cast<int>(parse_uint(*value, "--steps"));
    }
    if (auto value = parser.present<std::string>("--cfg-scale"))
    {
        config.cfg_scale = parse_float(*value, "--cfg-scale");
    }
    if (auto value = parser.present<std::string>("--flow-shift"))
    {
        config.flow_shift = parse_float(*value, "--flow-shift");
    }
    if (auto value = parser.present<std::string>("--init-noise"))
    {
        config.init_noise_path = *value;
    }
    if (auto value = parser.present<std::string>("--dump-dir"))
    {
        config.dump_dir = *value;
    }

    const std::filesystem::path output_dir = parser.get<std::string>("--output");
    std::filesystem::create_directories(output_dir);
    config.output_path = output_dir / "pixeldit.png";
    return config;
}

std::filesystem::path make_output_path(const PixelDiTConfig& config, unsigned int image_index)
{
    const std::string stem = config.output_path.stem().string();
    const std::string extension = config.output_path.extension().string();
    return config.output_path.parent_path() / (stem + "_" + std::to_string(image_index) + extension);
}

void save_image(const std::filesystem::path& output_path, const PixelDiTImage& image)
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
}  // namespace din::image_gen

int main(int argc, char* argv[])
{
    using namespace din::image_gen;
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    try
    {
        const PixelDiTConfig config = parse_args(argc, argv);
        std::cout << "Model dir: " << config.model_dir.string() << "\n"
                  << "Output: " << config.output_path.string() << "\n"
                  << "Prompt: " << config.prompt << "\n"
                  << "Seed:   " << config.seed << "\n"
                  << "Num images: " << config.num_images << "\n"
                  << "Processing: " << to_string(config.processing) << "\n"
                  << "Provider: " << to_string(config.provider) << "\n"
                  << "EP cache: " << config.ep_cache_dir.string() << "\n"
                  << "EP context dir: " << config.ep_context_dir.string() << "\n"
                  << std::endl;

        auto pipeline = CreatePixelDiTPipeline(config);
        pipeline->Initialize();

        for (unsigned int image_index = 0; image_index < config.num_images; ++image_index)
        {
            const unsigned int current_seed = config.seed + image_index;
            std::cout << "\n========== Image " << (image_index + 1) << "/" << config.num_images
                      << " (seed=" << current_seed << ") ==========" << std::endl;
            const auto image_start = std::chrono::steady_clock::now();
            PixelDiTImage image = pipeline->GenerateImage(current_seed);
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
