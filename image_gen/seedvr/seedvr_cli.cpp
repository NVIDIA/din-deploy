// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "io/image.h"
#include "seedvr_pipeline.h"
#include <argparse/argparse.hpp>
#include <lodepng.h>

namespace
{
struct CliConfig
    : din::seedvr::ModelConfig
    , din::seedvr::DynamicOptions
{
    std::filesystem::path input_path;
    std::filesystem::path output_path = "out/seedvr/seedvr-cpp";
    float scale = 2.0f;
    size_t video_frames = 1;
};
CliConfig ParseArgs(int argc, char* argv[])
{
    CliConfig config;
    argparse::ArgumentParser parser("din_seedvr2");
    parser.add_description(
        "Upscale a PNG or a directory of PNGs with SeedVR2 using ONNX Runtime, TensorRT RTX, and CUDA.");
    parser.add_argument("input").help(
        "Input PNG image or a video sequence of filename-sorted PNGs; folders require video exports.");
    parser.add_argument("--output")
        .default_value(config.output_path.string())
        .nargs(1)
        .metavar("DIR")
        .help("Output directory for PNG images named <input-stem>_seedvr.png.");
    parser.add_argument("--model-dir").default_value(config.model_dir.string()).nargs(1).metavar("DIR");
    parser.add_argument("--vae-backend").default_value(config.vae_backend).help("VAE backend: onnx or cudnn.");
    parser.add_argument("--vae-weights")
        .default_value(std::string{})
        .help("Native VAE directory produced by prepare_native_vae.py.");
    parser.add_argument("--vae-autotune")
        .default_value(std::string{"off"})
        .help("Native cuDNN graph-replay tuning: off or on.");
    parser.add_argument("--vae-tuning-cache")
        .default_value(config.vae_tuning_cache.string())
        .help("Persistent native cuDNN tuning records.");
    parser.add_argument("--vae-workspace-mib")
        .default_value(config.vae_workspace_mib)
        .scan<'i', int>()
        .help("Maximum workspace per cuDNN plan in MiB.");
    parser.add_argument("--vae-tuning-candidates")
        .default_value(config.vae_tuning_candidates)
        .scan<'i', int>()
        .help("Benchmark the first N supported cuDNN plans in heuristic order (default 5).");
    parser.add_argument("--transformer-precision")
        .default_value(config.transformer_precision)
        .nargs(1)
        .metavar("bf16|fp16|fp8_wo|nvfp4")
        .help("DiT precision; VAE and boundary tensors remain FP16.");
    parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string()).nargs(1).metavar("DIR");
    parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string()).nargs(1).metavar("DIR");
    parser.add_argument("--scale").default_value(std::to_string(config.scale)).nargs(1).metavar("FLOAT");
    parser.add_argument("--latent-noise-scale")
        .default_value(std::to_string(config.latent_noise_scale))
        .nargs(1)
        .metavar("FLOAT")
        .help("Conditioning-latent noise in [0, 1], default 0; try 0.05-0.15 to soften excessive detail.");
    parser.add_argument("--color-correction")
        .default_value(config.color_correction)
        .nargs(1)
        .metavar("lab|none")
        .help("Postprocess against the resized input: lab (default) or none.");
    parser.add_argument("--seed").default_value(std::to_string(config.seed)).nargs(1).metavar("N");
    parser.add_argument("--rng")
        .default_value(std::string{"cpu-std"})
        .nargs(1)
        .metavar("cpu-std|cuda-philox|cpu-philox")
        .help("Noise generator: cpu-std (default), cuda-philox (CUDA only), or cpu-philox (fixed H100 SXM5 ordering).");
    parser.add_argument("--ws")
        .default_value(std::string{"off"})
        .nargs(1)
        .metavar("BUDGET")
        .help("DiT TensorRT RTX weight-streaming budget, forwarded to ORT; off by default.");
    parser.add_argument("--vae-encoder-tile-size")
        .default_value(0)
        .scan<'i', int>()
        .help("Encoder tile size in upscaled pixels, multiple of 16; 0 disables tiling.");
    parser.add_argument("--vae-encoder-overlap")
        .default_value(64)
        .scan<'i', int>()
        .help("Encoder overlap in upscaled pixels, multiple of 16.");
    parser.add_argument("--vae-decoder-tile-size")
        .default_value(0)
        .scan<'i', int>()
        .help("Decoder tile size in upscaled pixels, multiple of 16; 0 disables tiling.");
    parser.add_argument("--vae-decoder-overlap")
        .default_value(64)
        .scan<'i', int>()
        .help("Decoder overlap in upscaled pixels, multiple of 16.");
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
    config.vae_encoder_tile_size = parser.get<int>("--vae-encoder-tile-size");
    config.vae_encoder_overlap = parser.get<int>("--vae-encoder-overlap");
    config.vae_decoder_tile_size = parser.get<int>("--vae-decoder-tile-size");
    config.vae_decoder_overlap = parser.get<int>("--vae-decoder-overlap");
    config.output_path = parser.get<std::string>("--output");
    config.model_dir = parser.get<std::string>("--model-dir");
    config.vae_backend = parser.get<std::string>("--vae-backend");
    config.vae_weights_dir = parser.get<std::string>("--vae-weights");
    const auto autotune = parser.get<std::string>("--vae-autotune");
    if (autotune != "off" && autotune != "on")
        throw std::invalid_argument("--vae-autotune must be off or on");
    config.vae_autotune = autotune == "on";
    config.vae_tuning_cache = parser.get<std::string>("--vae-tuning-cache");
    config.vae_workspace_mib = parser.get<int>("--vae-workspace-mib");
    config.vae_tuning_candidates = parser.get<int>("--vae-tuning-candidates");
    if (config.vae_tuning_candidates < 1)
        throw std::invalid_argument("--vae-tuning-candidates must be positive");
    if (config.vae_workspace_mib < 0)
        throw std::invalid_argument("--vae-workspace-mib must be nonnegative");
    config.transformer_precision = parser.get<std::string>("--transformer-precision");
    config.ep_cache_dir = parser.get<std::string>("--ep-cache");
    config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
    config.scale = std::stof(parser.get<std::string>("--scale"));
    const auto latent_noise = parser.get<std::string>("--latent-noise-scale");
    size_t latent_noise_parsed = 0;
    config.latent_noise_scale = std::stof(latent_noise, &latent_noise_parsed);
    if (latent_noise_parsed != latent_noise.size())
        throw std::invalid_argument("--latent-noise-scale must be a finite number in [0, 1]");
    config.color_correction = parser.get<std::string>("--color-correction");
    if (config.color_correction != "lab" && config.color_correction != "none")
    {
        throw std::invalid_argument("--color-correction must be lab or none");
    }
    const std::string seed = parser.get<std::string>("--seed");
    size_t parsed = 0;
    const unsigned long parsed_seed = std::stoul(seed, &parsed, 10);
    if (parsed != seed.size())
    {
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

bool IsPng(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character)
                   {
                       return static_cast<char>(std::tolower(character));
                   });
    return extension == ".png";
}

std::pair<size_t, size_t> ReadDimensions(const std::filesystem::path& path)
{
    std::vector<unsigned char> rgba;
    unsigned width = 0, height = 0;
    const auto error = lodepng::decode(rgba, width, height, path.string());
    if (error)
        throw std::runtime_error("Failed to decode " + path.string() + ": " + lodepng_error_text(error));
    return {height, width};
}

void LoadImage(const std::filesystem::path& path, Ort::Value& image)
{
    std::vector<unsigned char> rgba;
    unsigned width = 0, height = 0;
    const auto error = lodepng::decode(rgba, width, height, path.string());
    if (error)
        throw std::runtime_error("Failed to decode " + path.string() + ": " + lodepng_error_text(error));
    const auto shape = image.GetTensorTypeAndShapeInfo().GetShape();
    if (shape[1] != height || shape[2] != width)
        throw std::invalid_argument("Input dimensions changed: " + path.string());
    float* dst = image.GetTensorMutableData<float>();
    const size_t plane = size_t(height) * width;
    for (size_t p = 0; p < plane; ++p)
        for (size_t c = 0; c < 3; ++c)
            dst[c * plane + p] = static_cast<float>(rgba[p * 4 + c]) / 255.0f;
}

template <class Pipeline>
void Process(Pipeline& pipeline, const CliConfig& config, const std::vector<std::filesystem::path>& files)
{
    using namespace din::seedvr;
    const size_t frames = config.video_frames;
    std::vector<Ort::Value> inputs, outputs;
    inputs.reserve(frames);
    outputs.reserve(frames);
    for (size_t t = 0; t < frames; ++t)
    {
        inputs.push_back(pipeline.CreateImage(ImageRole::Input));
        outputs.push_back(pipeline.CreateImage(ImageRole::Output));
    }
    const auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const size_t oh = output_shape[1], ow = output_shape[2];
    std::vector<float> bytes(3 * oh * ow);
    for (size_t start = 0; start < files.size();)
    {
        const size_t end = std::min(start + frames, files.size());
        for (size_t t = 0; t < frames; ++t)
            LoadImage(files[std::min(start + t, end - 1)], inputs[t]);
        if constexpr (std::is_same_v<Pipeline, ImagePipeline>)
            pipeline.RunImage(inputs[0], outputs[0]);
        else
        {
            std::cout << "Video clip " << start + 1 << ".." << end << '/' << files.size() << " (model frames=" << frames
                      << ")" << std::endl;
            pipeline.RunChunk(inputs, outputs);
        }
        for (size_t t = start ? 1 : 0; t < end - start; ++t)
        {
            const auto destination = config.output_path / (files[start + t].stem().string() + "_seedvr.png");
            const float* pixels = outputs[t].template GetTensorData<float>();
            std::transform(pixels, pixels + bytes.size(), bytes.begin(),
                           [](float x)
                           {
                               return x * 255.0f;
                           });
            if (!din::io::SaveRgbFloatImage(destination.string(), bytes.data(), oh, ow,
                                            din::io::ImageValueRange::ZeroTo255))
                throw std::runtime_error("Failed to save " + destination.string());
            std::cout << "Saved " << destination.string() << std::endl;
        }
        if (end == files.size())
            break;
        start += frames - 1;
    }
}
}  // namespace

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const auto start = std::chrono::steady_clock::now();
        CliConfig config = ParseArgs(argc, argv);
        if (!std::isfinite(config.scale) || config.scale <= 0)
            throw std::invalid_argument("Scale must be finite and positive");
        const bool video = std::filesystem::is_directory(config.input_path);
        std::vector<std::filesystem::path> files;
        if (video)
        {
            for (const auto& entry : std::filesystem::directory_iterator(config.input_path))
                if (entry.is_regular_file() && IsPng(entry.path()))
                    files.push_back(entry.path());
            std::sort(files.begin(), files.end());
            if (files.empty())
                throw std::runtime_error("No PNG images found in " + config.input_path.string());
            std::ifstream manifest(config.model_dir / "seedvr2_video_frames.txt");
            if (!(manifest >> config.video_frames) || config.video_frames < 5 || (config.video_frames - 1) % 4)
                throw std::invalid_argument("Folder input requires video exports with T=4n+1, T >= 5");
        }
        else
        {
            if (!std::filesystem::is_regular_file(config.input_path))
                throw std::runtime_error("Input is not a file or directory");
            files.push_back(config.input_path);
        }
        std::filesystem::create_directories(config.output_path);
        if (video && std::filesystem::equivalent(config.input_path, config.output_path))
            throw std::invalid_argument("Output directory must differ from input directory");
        const auto [height, width] = ReadDimensions(files.front());
        for (const auto& file : files)
            if (ReadDimensions(file) != std::pair{height, width})
                throw std::invalid_argument("All sequence frames must have the same dimensions: " + file.string());
        const double th = std::nearbyint(height * config.scale), tw = std::nearbyint(width * config.scale);
        if (!std::isfinite(th) || !std::isfinite(tw) || th > 3840 || tw > 3840)
            throw std::invalid_argument("Scaled dimensions exceed profile limits");
        const din::seedvr::ImageShape shape{height, width, std::max<size_t>(2, static_cast<size_t>(th)),
                                            std::max<size_t>(2, static_cast<size_t>(tw))};
        if (video)
        {
            din::seedvr::VideoPipeline pipeline(config, {shape, config.video_frames}, config);
            Process(pipeline, config, files);
        }
        else
        {
            din::seedvr::ImagePipeline pipeline(config, shape, config);
            Process(pipeline, config, files);
        }
        std::cout << "Completed " << files.size() << '/' << files.size() << " images\nCLI wall: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds"
                  << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
