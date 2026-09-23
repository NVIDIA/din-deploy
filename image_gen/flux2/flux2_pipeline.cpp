// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <stdexcept>

#include "flux2_runtime_context.h"

namespace
{
Flux2RuntimeContext& GetProcessRuntime(Flux2ExecutionProvider provider)
{
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "Flux2"};
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
