// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sam2_io.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "lodepng.h"
#include "nvtx_helper.h"
#include <nlohmann/json.hpp>

namespace din::sam2
{

namespace fs = std::filesystem;

std::vector<int64_t> LoadRequiredShapeArray(const nlohmann::json& object, const char* key)
{
    const auto& value = object.at(key);
    if (!value.is_array())
    {
        throw std::runtime_error(std::string("metadata shape must be an array: ") + key);
    }
    std::vector<int64_t> shape;
    shape.reserve(value.size());
    for (const auto& entry : value)
    {
        shape.push_back(entry.get<int64_t>());
    }
    return shape;
}

std::vector<int64_t> LoadShapeArrayOrDefault(const nlohmann::json* object, const char* key,
                                             std::vector<int64_t> fallback)
{
    if (object == nullptr || !object->contains(key))
    {
        return fallback;
    }
    return LoadRequiredShapeArray(*object, key);
}

Sam2Config LoadSam2Config(const fs::path& metadata_path)
{
    std::ifstream stream(metadata_path);
    if (!stream)
    {
        throw std::runtime_error("failed to open SAM2 metadata: " + metadata_path.string());
    }
    const auto metadata = nlohmann::json::parse(stream);
    Sam2Config config;
    config.image_size = metadata.at("image_size").get<int64_t>();
    config.max_points = metadata.at("max_points").get<int64_t>();
    config.max_memory_frames = metadata.at("max_memory_frames").get<int64_t>();
    config.hidden_dim = metadata.at("hidden_dim").get<int64_t>();
    config.memory_dim = metadata.at("mem_dim").get<int64_t>();
    config.max_input_size = metadata.at("max_input_size").get<int64_t>();
    config.dtype = metadata.at("dtype").get<std::string>();
    config.spatial_contract = metadata.at("spatial_contract").get<std::string>();
    config.prompt_optimization_shapes = {
        .point_coords = {1, config.max_points, 2},
        .point_labels = {1, config.max_points},
        .input_masks = {1, 1, config.low_res_mask_size(), config.low_res_mask_size()},
        .has_input_masks = {1, 1, 1, 1},
    };
    config.decoder_optimization_shapes = {
        .image_features_0 = {1, 32, config.low_res_mask_size(), config.low_res_mask_size()},
        .image_features_1 = {1, 64, config.image_size / 8, config.image_size / 8},
    };

    const auto* prompt = metadata.contains("prompt_optimization_shapes")
                             ? &metadata.at("prompt_optimization_shapes")
                             : (metadata.contains("prompt_contract") ? &metadata.at("prompt_contract") : nullptr);
    config.prompt_optimization_shapes.point_coords =
        LoadShapeArrayOrDefault(prompt, "point_coords", config.prompt_optimization_shapes.point_coords);
    config.prompt_optimization_shapes.point_labels =
        LoadShapeArrayOrDefault(prompt, "point_labels", config.prompt_optimization_shapes.point_labels);
    config.prompt_optimization_shapes.input_masks =
        LoadShapeArrayOrDefault(prompt, "input_masks", config.prompt_optimization_shapes.input_masks);
    config.prompt_optimization_shapes.has_input_masks =
        LoadShapeArrayOrDefault(prompt, "has_input_masks", config.prompt_optimization_shapes.has_input_masks);

    const auto* decoder = metadata.contains("decoder_optimization_shapes")
                              ? &metadata.at("decoder_optimization_shapes")
                              : (metadata.contains("decoder_contract") ? &metadata.at("decoder_contract") : nullptr);
    config.decoder_optimization_shapes.image_features_0 =
        LoadShapeArrayOrDefault(decoder, "image_features_0", config.decoder_optimization_shapes.image_features_0);
    config.decoder_optimization_shapes.image_features_1 =
        LoadShapeArrayOrDefault(decoder, "image_features_1", config.decoder_optimization_shapes.image_features_1);

    return config;
}

Image LoadPng(const fs::path& path)
{
    DIN_NVTX_FUNC_RANGE();
    Image image;
    const unsigned error = lodepng::decode(image.rgb, image.width, image.height, path.string(), LCT_RGB, 8);
    if (error != 0)
    {
        throw std::runtime_error("failed to load PNG " + path.string() + ": " + lodepng_error_text(error));
    }
    return image;
}

void SaveRgbPng(const fs::path& path, const uint8_t* rgb, uint32_t width, uint32_t height)
{
    DIN_NVTX_FUNC_RANGE();
    if (!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
    const unsigned error = lodepng::encode(path.string(), rgb, width, height, LCT_RGB, 8);
    if (error != 0)
    {
        throw std::runtime_error("failed to save PNG " + path.string() + ": " + lodepng_error_text(error));
    }
}

Prompt LoadPrompt(const fs::path& path, uint32_t image_height, uint32_t image_width, int64_t image_size)
{
    DIN_NVTX_FUNC_RANGE();
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open prompt JSON: " + path.string());
    }
    const auto payload = nlohmann::json::parse(stream);
    const auto& record = payload.contains("prompts") ? payload.at("prompts").at(0) : payload;
    Prompt prompt;
    if (record.contains("box") && !record.at("box").is_null())
    {
        const auto box = record.at("box");
        prompt.coords.insert(
            prompt.coords.end(),
            {
                box.at(0).get<float>() / static_cast<float>(image_width) * static_cast<float>(image_size),
                box.at(1).get<float>() / static_cast<float>(image_height) * static_cast<float>(image_size),
                box.at(2).get<float>() / static_cast<float>(image_width) * static_cast<float>(image_size),
                box.at(3).get<float>() / static_cast<float>(image_height) * static_cast<float>(image_size),
            });
        prompt.labels.insert(prompt.labels.end(), {2, 3});
    }
    const auto points = record.value("points", nlohmann::json::array());
    const auto labels = record.value("labels", nlohmann::json::array());
    for (size_t i = 0; i < points.size(); ++i)
    {
        prompt.coords.push_back(points.at(i).at(0).get<float>() / static_cast<float>(image_width) *
                                static_cast<float>(image_size));
        prompt.coords.push_back(points.at(i).at(1).get<float>() / static_cast<float>(image_height) *
                                static_cast<float>(image_size));
        prompt.labels.push_back(labels.at(i).get<int32_t>());
    }
    if (prompt.labels.empty())
    {
        prompt.coords = {0.0f, 0.0f};
        prompt.labels = {-1};
    }
    return prompt;
}

std::vector<fs::path> ListFrames(const fs::path& dir)
{
    std::vector<fs::path> frames;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
        {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::vector<fs::path> ResolveInputFrames(const fs::path& input, size_t max_frames)
{
    std::vector<fs::path> frames;
    if (fs::is_directory(input))
    {
        frames = ListFrames(input);
        if (frames.empty())
        {
            throw std::runtime_error("no input frames found in " + input.string());
        }
    }
    else if (fs::is_regular_file(input))
    {
        frames.push_back(input);
    }
    else
    {
        throw std::runtime_error("missing input image or frames directory: " + input.string());
    }
    if (max_frames > 0 && frames.size() > max_frames)
    {
        frames.resize(max_frames);
    }
    return frames;
}

Prompt EmptyPropagatePrompt()
{
    return {.coords = {0.0f, 0.0f}, .labels = {-1}};
}

std::string FrameMaskName(size_t index, const char* suffix)
{
    std::ostringstream stream;
    stream << "frame_" << std::setw(6) << std::setfill('0') << index << suffix;
    return stream.str();
}

namespace detail
{

AsyncMaskWriter::AsyncMaskWriter(bool enabled)
    : enabled_(enabled)
{
    if (enabled_)
    {
        worker_ = std::thread(
            [this]
            {
                Run();
            });
    }
}

AsyncMaskWriter::~AsyncMaskWriter()
{
    try
    {
        Finish();
    }
    catch (...)
    {
    }
}

void AsyncMaskWriter::EnqueueRgb(const fs::path& path, const uint8_t* rgb, uint32_t width, uint32_t height)
{
    if (!enabled_)
    {
        SaveRgbPng(path, rgb, width, height);
        return;
    }

    Job job;
    job.path = path;
    job.width = width;
    job.height = height;
    job.rgb.assign(rgb, rgb + static_cast<size_t>(width) * height * 3);
    {
        std::unique_lock lock(mutex_);
        ThrowIfWorkerFailed();
        space_cv_.wait(lock,
                       [this]
                       {
                           return stop_ || jobs_.size() < kMaxQueuedJobs;
                       });
        ThrowIfWorkerFailed();
        if (stop_)
        {
            throw std::runtime_error("mask writer was stopped before enqueue");
        }
        jobs_.push_back(std::move(job));
    }
    work_cv_.notify_one();
}

void AsyncMaskWriter::Finish()
{
    if (!enabled_)
    {
        return;
    }
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_one();
    space_cv_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }
    ThrowIfWorkerFailed();
}

void AsyncMaskWriter::ThrowIfWorkerFailed() const
{
    if (exception_)
    {
        std::rethrow_exception(exception_);
    }
}

void AsyncMaskWriter::Run()
{
    while (true)
    {
        Job job;
        {
            std::unique_lock lock(mutex_);
            work_cv_.wait(lock,
                          [this]
                          {
                              return stop_ || !jobs_.empty();
                          });
            if (jobs_.empty())
            {
                if (stop_)
                {
                    return;
                }
                continue;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        space_cv_.notify_one();

        try
        {
            din::common::nvtx_scoped_range range{"sam2_async_mask_write"};
            SaveRgbPng(job.path, job.rgb.data(), job.width, job.height);
        }
        catch (...)
        {
            {
                std::lock_guard lock(mutex_);
                if (!exception_)
                {
                    exception_ = std::current_exception();
                }
                stop_ = true;
            }
            space_cv_.notify_all();
            work_cv_.notify_all();
        }
    }
}

}  // namespace detail

}  // namespace din::sam2
