// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <stdexcept>
#include <string>

#include "pixeldit.h"
#include <nlohmann/json.hpp>

namespace din::image_gen
{
namespace
{

// Reads object[key] into value when present, otherwise records `label` as a default that was used.
template <typename T>
void ReadField(const nlohmann::json& object, const char* key, T& value, const std::string& label,
               PixelDiTPipelineConfig& config)
{
    if (object.is_object() && object.contains(key) && !object[key].is_null())
    {
        value = object[key].get<T>();
    }
    else
    {
        config.defaults_used.push_back(label);
    }
}

}  // namespace

PixelDiTPipelineConfig LoadPixelDiTPipelineConfig(const std::filesystem::path& path)
{
    PixelDiTPipelineConfig config;
    nlohmann::json root = nlohmann::json::object();
    std::ifstream stream(path);
    if (stream)
    {
        root = nlohmann::json::parse(stream);
        config.file_found = true;
    }

    const nlohmann::json empty = nlohmann::json::object();
    const nlohmann::json& text = root.contains("text") ? root["text"] : empty;
    const nlohmann::json& sampler = root.contains("sampler") ? root["sampler"] : empty;

    ReadField(root, "height", config.height, "height", config);
    ReadField(root, "width", config.width, "width", config);
    ReadField(root, "dit_batch", config.dit_batch, "dit_batch", config);
    ReadField(root, "timestep_scale", config.timestep_scale, "timestep_scale", config);

    ReadField(text, "chi_prompt", config.chi_prompt, "text.chi_prompt", config);
    ReadField(text, "text_seq_len", config.text_seq_len, "text.text_seq_len", config);
    ReadField(text, "txt_max_length", config.txt_max_length, "text.txt_max_length", config);
    ReadField(text, "bos_token_id", config.bos_token_id, "text.bos_token_id", config);
    ReadField(text, "pad_token_id", config.pad_token_id, "text.pad_token_id", config);
    ReadField(text, "positive_select_index", config.positive_select_index, "text.positive_select_index", config);
    ReadField(text, "negative_select_index", config.negative_select_index, "text.negative_select_index", config);

    ReadField(sampler, "steps", config.steps, "sampler.steps", config);
    ReadField(sampler, "cfg_scale", config.cfg_scale, "sampler.cfg_scale", config);
    ReadField(sampler, "flow_shift", config.flow_shift, "sampler.flow_shift", config);
    ReadField(sampler, "negative_prompt", config.negative_prompt, "sampler.negative_prompt", config);
    ReadField(sampler, "t_start", config.t_start, "sampler.t_start", config);
    ReadField(sampler, "t_end", config.t_end, "sampler.t_end", config);
    ReadField(sampler, "order", config.order, "sampler.order", config);

    if (config.positive_select_index.empty())
    {
        config.positive_select_index = DefaultPositiveSelectIndex(config.text_seq_len, config.txt_max_length);
    }
    if (config.negative_select_index.empty())
    {
        config.negative_select_index = DefaultNegativeSelectIndex(config.txt_max_length);
    }

    const auto check_select = [&](const std::vector<int64_t>& index, const char* name)
    {
        if (static_cast<int64_t>(index.size()) != config.txt_max_length)
        {
            throw std::runtime_error(std::string(name) + " has " + std::to_string(index.size()) +
                                     " entries, expected txt_max_length=" + std::to_string(config.txt_max_length));
        }
        for (const int64_t i : index)
        {
            if (i < 0 || i >= config.text_seq_len)
            {
                throw std::runtime_error(std::string(name) + " entry " + std::to_string(i) + " is outside [0, " +
                                         std::to_string(config.text_seq_len) + ")");
            }
        }
    };
    check_select(config.positive_select_index, "positive_select_index");
    check_select(config.negative_select_index, "negative_select_index");
    if (config.dit_batch != 2)
    {
        throw std::runtime_error("Only dit_batch=2 ([negative, positive]) is supported, got " +
                                 std::to_string(config.dit_batch));
    }
    if (config.order != 2)
    {
        throw std::runtime_error("Only the 2nd-order DPM-Solver++ is supported, got order=" +
                                 std::to_string(config.order));
    }
    return config;
}

}  // namespace din::image_gen
