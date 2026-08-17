// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>

#include "whisper_kernels.h"

namespace din::asr::whisper
{
namespace
{

constexpr int kThreads = 256;
constexpr float kNegInf = -3.402823466e+38F;  // -FLT_MAX

__device__ inline float ToFloat(float v)
{
    return v;
}
__device__ inline float ToFloat(__half v)
{
    return __half2float(v);
}

// Single-block argmax over logits[lower, upper). Threads grid-stride the range,
// then reduce (value, index) pairs in shared memory. Ties resolve to the lower
// index, matching the CPU reference. Logits ~50k elements fit one block easily.
template <typename T>
__global__ void ArgmaxKernel(const T* logits, int64_t lower, int64_t upper, int32_t* out_index)
{
    __shared__ float s_val[kThreads];
    __shared__ int64_t s_idx[kThreads];

    float best = kNegInf;
    int64_t best_idx = lower;
    for (int64_t i = lower + threadIdx.x; i < upper; i += blockDim.x)
    {
        const float v = ToFloat(logits[i]);
        if (v > best)
        {
            best = v;
            best_idx = i;
        }
    }
    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const float other = s_val[threadIdx.x + stride];
            const int64_t other_idx = s_idx[threadIdx.x + stride];
            // Prefer the strictly larger value; on a tie keep the lower index.
            if (other > s_val[threadIdx.x] || (other == s_val[threadIdx.x] && other_idx < s_idx[threadIdx.x]))
            {
                s_val[threadIdx.x] = other;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *out_index = static_cast<int32_t>(s_idx[0]);
    }
}

}  // namespace

void launch_whisper_argmax(cudaStream_t stream, const void* logits, bool is_fp16, int64_t lower, int64_t upper,
                           int32_t* out_index)
{
    if (is_fp16)
    {
        ArgmaxKernel<__half><<<1, kThreads, 0, stream>>>(static_cast<const __half*>(logits), lower, upper, out_index);
    }
    else
    {
        ArgmaxKernel<float><<<1, kThreads, 0, stream>>>(static_cast<const float*>(logits), lower, upper, out_index);
    }
}

}  // namespace din::asr::whisper
