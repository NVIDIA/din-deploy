// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>

#include "seedvr_cuda_kernels.h"

namespace
{
    constexpr int kThreads = 256;

__global__ void PackCondition(const __half* encoded, __half* condition, size_t spatial_size)
    {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t element_count = 17 * spatial_size;
        if (index >= element_count)
        {
            return;
        }
        condition[index] = index < 16 * spatial_size ? encoded[index] : __float2half(1.0f);
    }

__global__ void EulerOneStep(const __half* latent, const __half* velocity, __half* sampled, size_t element_count)
    {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < element_count)
        {
            sampled[index] = __hsub(latent[index], velocity[index]);
        }
    }
} // namespace

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void* encoded, void* condition, size_t spatial_size)
{
    const size_t count = 17 * spatial_size;
    PackCondition<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half*>(encoded), static_cast<__half*>(condition), spatial_size);
}

void LaunchSeedVrEulerKernel(cudaStream_t stream, const void* latent, const void* velocity, void* sampled,
                             size_t element_count)
{
    EulerOneStep<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half*>(latent), static_cast<const __half*>(velocity), static_cast<__half*>(sampled),
        element_count);
}
