// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "flux2_models.h"

void ConfigureFlux2Models(Flux2Config& config, const std::filesystem::path& model_root,
                          const std::filesystem::path& cache_root, const std::filesystem::path& context_root)
{
    const auto root = model_root.empty() ? model_root : std::filesystem::weakly_canonical(model_root);
    std::string prefix;
    switch (config.processing)
    {
    case Flux2ProcessingBackend::Dx: prefix = "dx"; break;
    case Flux2ProcessingBackend::DxCig: prefix = "dx_cig"; break;
    case Flux2ProcessingBackend::Vk: prefix = "vk"; break;
    case Flux2ProcessingBackend::VkCig: prefix = "vk_cig"; break;
    default: break;
    }
    if (!prefix.empty()) prefix += '_';
    const std::string encoder = config.text_encoder.variant == Flux2TextEncoder::Qwen3_4B
                                    ? "text_encoder" : "text_encoder_translator";
    const std::string transformer = "transformer_" + config.transformer.precision;
    for (Flux2ModelConfig* model : {static_cast<Flux2ModelConfig*>(&config.text_encoder),
                                   static_cast<Flux2ModelConfig*>(&config.transformer),
                                   static_cast<Flux2ModelConfig*>(&config.vae)})
    {
        model->ep_cache_dir = std::filesystem::weakly_canonical(cache_root);
        model->ep_context_dir = std::filesystem::weakly_canonical(context_root);
    }
    config.text_encoder.cache_subpath = prefix + encoder;
    config.transformer.cache_subpath = prefix + transformer +
        (config.transformer.initial_weight_streaming_budget.empty() ? "" : "_ws");
    config.vae.cache_subpath = prefix + "vae_decoder";
    if (root.empty()) return; // The studio can start before a model root is selected.
    config.text_encoder.model_path = root / encoder / "model.onnx";
    config.text_encoder.tokenizer_dir = root / "tokenizer";
    config.transformer.model_path = root / transformer / "model.onnx";
    config.vae.model_path = root / "vae_decoder/model.onnx";
    config.vae.latent_stats_path = root / "vae_decoder/latent_stats.json";
}

std::unique_ptr<Ort::IoBinding> BindFlux2Model(Ort::Session& session,
    std::initializer_list<Flux2TensorBinding> inputs, std::initializer_list<Flux2TensorBinding> outputs)
{
    auto binding = std::make_unique<Ort::IoBinding>(session);
    Ort::AllocatorWithDefaultOptions allocator;
    auto bind = [&](std::initializer_list<Flux2TensorBinding> tensors, bool input)
    {
        const size_t count = input ? session.GetInputCount() : session.GetOutputCount();
        if (count != tensors.size()) throw std::runtime_error("Unexpected model input/output count");
        for (const auto& tensor : tensors)
        {
            size_t index = 0;
            for (; index < count; ++index)
            {
                const auto name = input ? session.GetInputNameAllocated(index, allocator)
                                        : session.GetOutputNameAllocated(index, allocator);
                if (std::string(name.get()) == tensor.name) break;
            }
            if (index == count) throw std::runtime_error(std::string("Missing tensor: ") + tensor.name);
            const auto type = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
            const auto info = type.GetTensorTypeAndShapeInfo();
            const auto actual = tensor.tensor->GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            const auto expected = actual.GetShape();
            if (info.GetElementType() != actual.GetElementType() || shape.size() != expected.size())
                throw std::runtime_error(std::string("Incompatible tensor type/rank: ") + tensor.name);
            for (size_t axis = 0; axis < shape.size(); ++axis)
                if (shape[axis] >= 0 && shape[axis] != expected[axis])
                    throw std::runtime_error(std::string("Incompatible tensor shape: ") + tensor.name);
            if (input) binding->BindInput(tensor.name, *tensor.tensor);
            else binding->BindOutput(tensor.name, *tensor.tensor);
        }
    };
    bind(inputs, true);
    bind(outputs, false);
    return binding;
}
