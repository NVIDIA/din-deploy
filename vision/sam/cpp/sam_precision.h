// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace din::sam::detail
{
    enum class ModelPrecision { Float32, Float16, BFloat16 };
}

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace din::sam::detail
{
    template <class F>
    void DispatchModelPrecision(ModelPrecision precision, F&& run)
    {
        switch (precision)
        {
        case ModelPrecision::Float16: run.template operator()<__half>();
            break;
        case ModelPrecision::BFloat16: run.template operator()<__nv_bfloat16>();
            break;
        default: run.template operator()<float>();
            break;
        }
    }
}
#endif
