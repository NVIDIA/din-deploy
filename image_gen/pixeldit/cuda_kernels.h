// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace din::image_gen
{

// velocity: [2, n] ([negative, positive]); sample: [n] at time t -> x0: [n]. See cfg_data_prediction.
void launch_pixeldit_cfg_x0_kernel(cudaStream_t stream, const float* velocity, const float* sample, float t,
                                   float cfg_scale, float* x0, size_t n);

// sample = ratio * sample - coeff * model_s. See dpm_first_order_update (t > 0).
void launch_pixeldit_dpm1_kernel(cudaStream_t stream, float* sample, const float* model_s, float ratio, float coeff,
                                 size_t n);

// See dpm_second_order_update.
void launch_pixeldit_dpm2_kernel(cudaStream_t stream, float* sample, const float* model_prev_1,
                                 const float* model_prev_0, float ratio, float coeff, float half_coeff, float inv_r0,
                                 size_t n);

}  // namespace din::image_gen
