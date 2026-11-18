// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include "sam3_nvtx.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "sam3_nvtx.h"

namespace din::sam3
{
    void SaveResult(const FrameResult& result, const std::filesystem::path& directory, uint64_t frame_index,
                    const RasterizedMasks* rasters)
    {
        TraceRange trace{"io.SaveResult"};
        const size_t pixels = size_t(result.width) * result.height;
        if (rasters)
        {
            if (rasters->width != result.width || rasters->height != result.height || rasters->ids.size() != result.
                instances.size() ||
                rasters->individual.size() != pixels * result.instances.size())
                throw std::invalid_argument("Mismatched rasterized masks");
            for (size_t i = 0; i < result.instances.size(); ++i)
                if (rasters->ids[i] != result.instances[i].id)
                    throw std::invalid_argument(
                        "Mismatched raster instance IDs");
        }
        std::filesystem::create_directories(directory);
        std::ostringstream prefix;
        prefix << std::setfill('0') << std::setw(6) << frame_index;
        size_t index = 0;
        for (const auto& instance : result.instances)
        {
            std::vector<uint8_t> mask(pixels);
            if (rasters)
                std::copy_n(rasters->individual.data() + index * pixels, pixels, mask.data());
            else
                for (size_t i = 0; i < pixels; ++i)
                {
                    const float value = instance.probabilities[i];
                    if (!std::isfinite(value) || value < 0 || value > 1)
                        throw std::runtime_error("Invalid mask probability");
                    mask[i] = value > .5f ? 255 : 0;
                }
            din::io::SaveGrayPng(directory / (prefix.str() + "_id" + std::to_string(instance.id) + ".png"),
                                 mask.data(), result.width, result.height);
            ++index;
        }
    }

    std::vector<uint32_t> FrameResult::Labels() const
    {
        if (!width || !height || width > 8192 || height > 8192 || instances.size() > 16)
            throw std::runtime_error("Invalid SAM3 result dimensions or capacity");
        std::vector<uint32_t> labels(size_t(width) * height, 0);
        std::vector<float> best(labels.size(), .5f);
        std::set<uint32_t> ids;
        for (const auto& instance : instances)
        {
            if (instance.id == 0 || !ids.insert(instance.id).second || instance.probabilities.size() != labels.size())
                throw std::runtime_error("Invalid SAM3 instance result");
            for (size_t i = 0; i < labels.size(); ++i)
            {
                float probability = instance.probabilities[i];
                if (!std::isfinite(probability) || probability < 0 || probability > 1)
                    throw std::runtime_error(
                        "Invalid mask probability");
                if (probability > best[i] || (probability == best[i] && probability > .5f && instance.id < labels[i]))
                {
                    best[i] = probability;
                    labels[i] = instance.id;
                }
            }
        }
        return labels;
    }

    std::vector<uint8_t> FrameResult::CombinedMask() const
    {
        const auto labels = Labels();
        std::vector<uint8_t> mask(labels.size());
        std::transform(labels.begin(), labels.end(), mask.begin(), [](uint32_t label) { return label ? 255 : 0; });
        return mask;
    }
}
