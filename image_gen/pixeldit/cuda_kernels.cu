// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <stdexcept>
#include <string>

#include "cuda_kernels.h"

namespace din::image_gen
{
namespace
{

constexpr unsigned int kThreadsPerBlock = 256;

void check_launch()
{
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

dim3 grid_for(size_t total_elements)
{
    const size_t blocks = (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > std::numeric_limits<unsigned int>::max())
    {
        throw std::runtime_error("CUDA grid is too large");
    }
    return dim3(static_cast<unsigned int>(blocks), 1, 1);
}

__global__ void pixeldit_cfg_x0_kernel(const float* velocity, const float* x, float t, float cfg_scale, float* x0,
                                       size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float one_minus_t = 1.0f - t;
    const float scaled_u = one_minus_t * velocity[i];
    const float noise_u = scaled_u + x[i];
    const float scaled_c = one_minus_t * velocity[n + i];
    const float noise_c = scaled_c + x[i];
    const float diff = noise_c - noise_u;
    const float guided_delta = cfg_scale * diff;
    const float noise = noise_u + guided_delta;
    const float t_noise = t * noise;
    const float numerator = x[i] - t_noise;
    x0[i] = numerator / one_minus_t;
}

__global__ void pixeldit_dpm1_kernel(float* x, const float* model_s, float ratio, float coeff, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float a = ratio * x[i];
    const float b = coeff * model_s[i];
    x[i] = a - b;
}

__global__ void pixeldit_dpm2_kernel(float* x, const float* model_prev_1, const float* model_prev_0, float ratio,
                                     float coeff, float half_coeff, float inv_r0, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float delta = model_prev_0[i] - model_prev_1[i];
    const float d1 = inv_r0 * delta;
    const float a = ratio * x[i];
    const float b = coeff * model_prev_0[i];
    const float c = half_coeff * d1;
    const float ab = a - b;
    x[i] = ab - c;
}

}  // namespace

void launch_pixeldit_cfg_x0_kernel(cudaStream_t stream, const float* velocity, const float* sample, float t,
                                   float cfg_scale, float* x0, size_t n)
{
    pixeldit_cfg_x0_kernel<<<grid_for(n), kThreadsPerBlock, 0, stream>>>(velocity, sample, t, cfg_scale, x0, n);
    check_launch();
}

void launch_pixeldit_dpm1_kernel(cudaStream_t stream, float* sample, const float* model_s, float ratio, float coeff,
                                 size_t n)
{
    pixeldit_dpm1_kernel<<<grid_for(n), kThreadsPerBlock, 0, stream>>>(sample, model_s, ratio, coeff, n);
    check_launch();
}

void launch_pixeldit_dpm2_kernel(cudaStream_t stream, float* sample, const float* model_prev_1,
                                 const float* model_prev_0, float ratio, float coeff, float half_coeff, float inv_r0,
                                 size_t n)
{
    pixeldit_dpm2_kernel<<<grid_for(n), kThreadsPerBlock, 0, stream>>>(sample, model_prev_1, model_prev_0, ratio,
                                                                        coeff, half_coeff, inv_r0, n);
    check_launch();
}

}  // namespace din::image_gen
