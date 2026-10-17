// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rng_cuda.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cuda_bf16.h>
#include <curand_kernel.h>

namespace din::common
{
namespace
{
constexpr unsigned int kThreadsPerBlock = 256;
void check_cuda(cudaError_t result)
{
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
// Match ATen DistributionTemplates.h: one Philox subsequence per thread,
// normal4 components spaced by the entire grid, then grid-stride by 4.
__global__ __launch_bounds__(256, 4) void normal_philox_kernel(
    uint64_t seed, size_t count, bool round_to_bfloat16, size_t channels, size_t spatial, float* output)
{
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t rounded = ((count - 1) / (stride * 4) + 1) * stride * 4;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread, 0, &state);
    for (size_t base = thread; base < rounded; base += stride * 4)
    {
        const float4 samples = curand_normal4(&state);
        const float values[] = {samples.x, samples.y, samples.z, samples.w};
#pragma unroll
        for (int component = 0; component < 4; ++component)
        {
            const size_t index = base + stride * component;
            if (index < count)
            {
                const size_t destination = channels == 0 ? index
                    : ((index / (channels * spatial)) * spatial + index % spatial) * channels
                        + (index / spatial) % channels;
                output[destination] = round_to_bfloat16
                    ? __bfloat162float(__float2bfloat16_rn(values[component])) : values[component];
            }
        }
    }
}

void generate_normal_cuda(cudaStream_t stream, float* output, size_t count, uint64_t seed,
                          bool round_to_bfloat16, size_t channels, size_t spatial)
{
    if (count == 0) return;
    if (output == nullptr || count > std::numeric_limits<size_t>::max() / 2)
        throw std::invalid_argument("CUDA normal generation requires a valid buffer and supported count");
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaGetDeviceProperties(&properties, device));
    // ATen caps the grid by the device's thread capacity, not kernel occupancy.
    const size_t capacity = static_cast<size_t>(properties.multiProcessorCount) *
                            (properties.maxThreadsPerMultiProcessor / kThreadsPerBlock);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(capacity, (count + kThreadsPerBlock - 1) / kThreadsPerBlock));
    normal_philox_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        seed, count, round_to_bfloat16, channels, spatial, output);
    check_cuda(cudaPeekAtLastError());
}

}

void GenerateNormalCuda(cudaStream_t stream, float* output, size_t count, uint64_t seed,
                        bool round_to_bfloat16)
{
    generate_normal_cuda(stream, output, count, seed, round_to_bfloat16, 0, 0);
}

void GenerateNormalCudaPacked(cudaStream_t stream, float* output, size_t count, uint64_t seed,
                             size_t channels, size_t spatial, bool round_to_bfloat16)
{
    if (channels == 0 || spatial == 0 || channels > std::numeric_limits<size_t>::max() / spatial ||
        count % (channels * spatial) != 0)
        throw std::invalid_argument("Packed CUDA normal dimensions must match the buffer");
    generate_normal_cuda(stream, output, count, seed, round_to_bfloat16, channels, spatial);
}

} // namespace din::common
