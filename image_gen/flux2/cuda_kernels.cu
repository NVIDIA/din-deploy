// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "cu_helper.h"
#include "cuda_kernels.h"

namespace
{
constexpr unsigned int kThreadsPerBlock = 256;

__global__ void flux_euler_kernel(float t_curr, float t_next, size_t total_elements, float* hidden_states,
                                  const float* transformer_output)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
    {
        return;
    }

    hidden_states[idx] += (t_next - t_curr) * transformer_output[idx];
}

__global__ void flux_postprocess_kernel(const float* input_data, float* output_data, const float* bn_mean,
                                        const float* bn_std, unsigned int channels, unsigned int latent_height,
                                        unsigned int latent_width, unsigned int patch_i, unsigned int patch_j,
                                        size_t total_elements)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
    {
        return;
    }

    const unsigned int c_out_count = channels / (patch_i * patch_j);
    const unsigned int h_out = latent_height * patch_i;
    const unsigned int w_out = latent_width * patch_j;
    (void)c_out_count;

    const unsigned int hw_out = h_out * w_out;
    const unsigned int c_out = static_cast<unsigned int>(idx / hw_out);
    const unsigned int rem = static_cast<unsigned int>(idx % hw_out);
    const unsigned int i_out = rem / w_out;
    const unsigned int j_out = rem % w_out;

    const unsigned int i = i_out / patch_i;
    const unsigned int p = i_out % patch_i;
    const unsigned int j = j_out / patch_j;
    const unsigned int q = j_out % patch_j;
    const unsigned int in_c = c_out * patch_i * patch_j + p * patch_j + q;

    const unsigned int hw = i * latent_width + j;
    output_data[idx] = input_data[static_cast<size_t>(hw) * channels + in_c] * bn_std[in_c] + bn_mean[in_c];
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
}  // namespace

void launch_flux_euler_kernel(cudaStream_t stream, float t_curr, float t_next, size_t total_elements,
                              float* hidden_states, const float* transformer_output)
{
    flux_euler_kernel<<<grid_for(total_elements), kThreadsPerBlock, 0, stream>>>(t_curr, t_next, total_elements,
                                                                                 hidden_states, transformer_output);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launch_flux_postprocess_kernel(cudaStream_t stream, const float* input_data, float* output_data,
                                    const float* bn_mean, const float* bn_std, unsigned int channels,
                                    unsigned int latent_height, unsigned int latent_width, unsigned int patch_i,
                                    unsigned int patch_j, size_t total_elements)
{
    flux_postprocess_kernel<<<grid_for(total_elements), kThreadsPerBlock, 0, stream>>>(
        input_data, output_data, bn_mean, bn_std, channels, latent_height, latent_width, patch_i, patch_j,
        total_elements);
    CUDA_CHECK(cudaPeekAtLastError());
}
