// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <stdexcept>

#include "pixeldit_runtime_context.h"

namespace din::image_gen
{
namespace
{

PixelDiTRuntimeContext& GetProcessRuntime()
{
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PixelDiT"};
    static PixelDiTRuntimeContext runtime{env};
    static std::once_flag register_trt_once;

    std::call_once(register_trt_once,
                   []
                   {
                       runtime.trt_device = din::common::RegisterTensorRTRTXProvider(runtime.env);
                   });
    return runtime;
}

}  // namespace

std::unique_ptr<PixelDiTProcessingPipeline> CreatePixelDiTPipeline(const PixelDiTConfig& config)
{
    if (config.provider != PixelDiTExecutionProvider::TrtRtx)
    {
        throw std::invalid_argument("PixelDiT supports only --provider trt-rtx");
    }
    PixelDiTRuntimeContext& runtime = GetProcessRuntime();

    if (config.processing == PixelDiTProcessingBackend::Cpu)
    {
        return CreatePixelDiTTrtPipeline(config, runtime);
    }

    throw std::invalid_argument("Unknown PixelDiT processing backend");
}

}  // namespace din::image_gen
