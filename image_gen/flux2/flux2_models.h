// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flux2_cli.h"
#include "ort_session.h"
#include <initializer_list>
#include <iostream>

struct Flux2TensorBinding
{
    const char* name;
    Ort::Value* tensor;
};

std::unique_ptr<Ort::IoBinding> BindFlux2Model(Ort::Session& session,
    std::initializer_list<Flux2TensorBinding> inputs, std::initializer_list<Flux2TensorBinding> outputs);

struct Flux2LoadedModels
{
    std::optional<Flux2TextEncoderConfig> text_encoder;
    std::optional<Flux2TransformerConfig> transformer;
    std::optional<Flux2VaeConfig> vae;
    bool Matches(const Flux2Config& config) const
    {
        return text_encoder == config.text_encoder && transformer == config.transformer && vae == config.vae;
    }
    bool Ready() const { return text_encoder && transformer && vae; }
};

// The caller drains device work before entering this function. Tensors and the
// externally owned stream remain alive across session replacement.
template<class Config, class Bind>
bool ReloadFlux2Stage(const char* name, const Config& requested, std::optional<Config>& loaded,
                     std::unique_ptr<din::common::OrtRunner>& runner, std::unique_ptr<Ort::IoBinding>& binding,
                     Ort::Env& env, const char* provider, Ort::SyncStream* stream,
                     din::common::ModelProfile profile, Bind&& bind)
{
    if (runner && binding && loaded == requested)
    {
        std::cout << "Reusing " << name << '\n';
        return false;
    }
    std::cout << "Loading " << name << ": " << requested.model_path.string() << '\n';
    binding.reset();
    runner.reset();
    loaded.reset();
    try
    {
        profile.cache_subpath = requested.cache_subpath;
        runner = std::make_unique<din::common::OrtRunner>(env, requested.model_path.string(), provider,
            requested.ep_cache_dir.string(), din::common::EpContextOptions{requested.ep_context_dir.string()},
            profile, stream);
        binding = bind(runner->session);
        loaded = requested;
    }
    catch (const std::exception& error)
    {
        binding.reset();
        runner.reset();
        loaded.reset();
        throw Flux2StageLoadError(std::string("Failed to load ") + name + " (" +
                                 requested.model_path.string() + "): " + error.what());
    }
    return true;
}
