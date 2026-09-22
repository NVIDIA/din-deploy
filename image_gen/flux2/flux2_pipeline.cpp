// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "flux2_runtime_context.h"

namespace
{
Flux2RuntimeContext& GetProcessRuntime(Flux2ExecutionProvider provider)
{
    static Ort::Env env{
        std::getenv("DIN_FLUX2_VERBOSE") ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING,
        "Flux2"
    };
    static Flux2RuntimeContext runtime{env};
    static std::once_flag register_trt_once;

    if (provider == Flux2ExecutionProvider::TrtRtx)
    {
        std::call_once(register_trt_once,
                       []
                       {
                           runtime.trt_device = din::common::RegisterTensorRTRTXProvider(runtime.env);
                       });
    }
    return runtime;
}
}  // namespace

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2Pipeline(const Flux2Config& config)
{
    ValidateFlux2Config(config);
    Flux2RuntimeContext& runtime = GetProcessRuntime(config.provider);

    if (config.processing == Flux2ProcessingBackend::Cpu || config.processing == Flux2ProcessingBackend::Cuda)
    {
        return CreateFlux2CudaPipeline(config, runtime);
    }

    if (config.processing == Flux2ProcessingBackend::Dx || config.processing == Flux2ProcessingBackend::DxCig)
    {
#if defined(DIN_FLUX2_BUILD_DX)
        return CreateFlux2DxPipeline(config, runtime);
#else
        throw std::runtime_error("The DirectX backend was not built");
#endif
    }

    if (config.processing == Flux2ProcessingBackend::Vk || config.processing == Flux2ProcessingBackend::VkCig)
    {
#if defined(DIN_FLUX2_BUILD_VK)
        return CreateFlux2VkPipeline(config, runtime);
#else
        throw std::runtime_error("The Vulkan backend was not built");
#endif
    }

    throw std::invalid_argument("Unknown Flux2 processing backend");
}

bool IsFlux2BackendAvailable(Flux2ProcessingBackend backend)
{
    switch (backend)
    {
    case Flux2ProcessingBackend::Cpu:
    case Flux2ProcessingBackend::Cuda:
        return true;
#if defined(DIN_FLUX2_BUILD_DX)
    case Flux2ProcessingBackend::Dx:
    case Flux2ProcessingBackend::DxCig:
        return true;
#endif
#if defined(DIN_FLUX2_BUILD_VK)
    case Flux2ProcessingBackend::Vk:
    case Flux2ProcessingBackend::VkCig:
        return true;
#endif
    default:
        return false;
    }
}

void ValidateFlux2Config(const Flux2Config& config)
{
    if (!IsFlux2BackendAvailable(config.processing))
        throw std::invalid_argument("Selected processing backend was not built");
    if (config.provider == Flux2ExecutionProvider::Cpu && config.processing != Flux2ProcessingBackend::Cpu)
        throw std::invalid_argument("CPU execution requires CPU processing");
    if (config.steps < 1 || config.steps > 50)
        throw std::invalid_argument("Denoise steps must be in [1, 50]");
    if (config.num_images == 0)
        throw std::invalid_argument("Image count must be positive");
    if (config.precision != "bf16" && config.precision != "fp16" && config.precision != "fp8" &&
        config.precision != "nvfp4")
        throw std::invalid_argument("Precision must be bf16, fp16, fp8, or nvfp4");
    if (!config.weight_streaming_budget.empty())
    {
        if (config.provider != Flux2ExecutionProvider::TrtRtx)
            throw std::invalid_argument("Weight streaming requires TensorRT RTX");
        const auto& budget = config.weight_streaming_budget;
        if (budget == "-1")
            return;
        size_t parsed = 0;
        const int value = std::stoi(budget, &parsed);
        if (parsed + 1 != budget.size() || budget.back() != '%' || value < 0 || value > 100)
            throw std::invalid_argument("Weight streaming budget must be 0% through 100%");
    }
}
