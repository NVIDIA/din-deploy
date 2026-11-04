// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include "sam3_nvtx.h"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <lodepng.h>
#include <nlohmann/json.hpp>

namespace din::sam3 {
Image LoadPng(const std::filesystem::path& path) {

    Image result;
    const auto status = lodepng::decode(result.rgb, result.width, result.height, path.string(), LCT_RGB, 8);
    if (status) throw std::runtime_error("Cannot decode " + path.string() + ": " + lodepng_error_text(status));
    return result;
}
void SaveResult(const FrameResult& result, const std::filesystem::path& directory, uint64_t frame_index, const RasterizedMasks* rasters) {

    const size_t pixels = size_t(result.width) * result.height;
    if (rasters) {
        if (rasters->width != result.width || rasters->height != result.height || rasters->ids.size() != result.instances.size() ||
            rasters->labels.size() != pixels || rasters->combined.size() != pixels || rasters->individual.size() != pixels * result.instances.size())
            throw std::invalid_argument("Mismatched rasterized masks");
        for (size_t i = 0; i < result.instances.size(); ++i)
            if (rasters->ids[i] != result.instances[i].id) throw std::invalid_argument("Mismatched raster instance IDs");
    }
    std::filesystem::create_directories(directory);
    std::ostringstream prefix; prefix << std::setfill('0') << std::setw(6) << frame_index;
    auto save_mask = [&](const std::string& name, const std::vector<uint8_t>& mask) {

        const auto error = lodepng::encode((directory / name).string(), mask.data(), result.width, result.height, LCT_GREY, 8);
        if (error) throw std::runtime_error(lodepng_error_text(error));
    };
    save_mask(prefix.str() + "_combined.png", rasters ? rasters->combined : result.CombinedMask());
    auto records = nlohmann::json::array();
    for (const auto& instance : result.instances) {
        std::vector<uint8_t> mask(instance.probabilities.size());
        if (rasters) std::copy_n(rasters->individual.data() + records.size() * pixels, pixels, mask.data());
        else for (size_t i = 0; i < mask.size(); ++i) mask[i] = instance.probabilities[i] > .5f ? 255 : 0;
        const auto name = prefix.str() + "_id" + std::to_string(instance.id) + ".png";
        save_mask(name, mask);
        records.push_back({{"id", instance.id}, {"score", instance.score}, {"box", instance.box}, {"mask", name}});
    }
    {
    std::ofstream json(directory / (prefix.str() + ".json"));
    json.exceptions(std::ios::failbit | std::ios::badbit);
    json << records.dump(2) << '\n';
    }
    const auto labels = rasters ? rasters->labels : result.Labels();

    static_assert(std::endian::native == std::endian::little);
    std::string header = "{'descr': '<u4', 'fortran_order': False, 'shape': (" + std::to_string(result.height) + ", " + std::to_string(result.width) + "), }";
    header.append((64 - (10 + header.size() + 1) % 64) % 64, ' '); header += '\n';
    std::ofstream label_file(directory / (prefix.str() + "_labels.npy"), std::ios::binary);
    label_file.exceptions(std::ios::failbit | std::ios::badbit);
    label_file.write("\x93NUMPY\x01\x00", 8);
    const uint16_t size = static_cast<uint16_t>(header.size());
    label_file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    label_file.write(header.data(), static_cast<std::streamsize>(header.size()));
    label_file.write(reinterpret_cast<const char*>(labels.data()), static_cast<std::streamsize>(labels.size() * sizeof(uint32_t)));
}
}
