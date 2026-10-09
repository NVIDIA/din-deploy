// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ort_session.h"
#include "pixeldit_cli.h"

namespace din::image_gen
{

struct PixelDiTRuntimeContext
{
    explicit PixelDiTRuntimeContext(Ort::Env& environment)
        : env(environment)
    {
    }

    Ort::Env& env;
    Ort::ConstEpDevice trt_device{};
};

std::unique_ptr<PixelDiTProcessingPipeline> CreatePixelDiTTrtPipeline(const PixelDiTConfig& config,
                                                                       PixelDiTRuntimeContext& runtime);

}  // namespace din::image_gen
