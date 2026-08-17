// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flux2_cli.h"
#include "ort_session.h"

struct Flux2RuntimeContext
{
    explicit Flux2RuntimeContext(Ort::Env& environment)
        : env(environment)
    {
    }

    Ort::Env& env;
    Ort::ConstEpDevice trt_device{};
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2CudaPipeline(const Flux2Config& config,
                                                                 Flux2RuntimeContext& runtime);
std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2DxPipeline(const Flux2Config& config, Flux2RuntimeContext& runtime);
std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2VkPipeline(const Flux2Config& config, Flux2RuntimeContext& runtime);
