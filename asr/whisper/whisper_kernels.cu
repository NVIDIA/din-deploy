// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>

#include <climits>
#include <cmath>

#include "whisper_kernels.h"
#include <math_constants.h>

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

struct Reduction
{
    float maximum;
    int index;
    double sum;
};
struct Partial
{
    Reduction text, timestamp;
};
static_assert(sizeof(Partial) == 4 * sizeof(double));

__device__ Reduction Empty()
{
    return {-CUDART_INF_F, INT_MAX, 0.0};
}

__device__ Reduction Merge(Reduction a, Reduction b)
{
    if (a.sum == 0)
        return b;
    if (b.sum == 0)
        return a;
    if (b.maximum > a.maximum || (b.maximum == a.maximum && b.index < a.index))
    {
        const Reduction tmp = a;
        a = b;
        b = tmp;
    }
    a.sum += b.sum * exp(double(b.maximum) - a.maximum);
    return a;
}

__device__ void Reduce(Partial* values)
{
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            values[threadIdx.x].text = Merge(values[threadIdx.x].text, values[threadIdx.x + stride].text);
            values[threadIdx.x].timestamp =
                Merge(values[threadIdx.x].timestamp, values[threadIdx.x + stride].timestamp);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void FilterReduce(const T* logits, int vocab, const uint8_t* suppressed, TimestampFilter f, bool raw,
                             Partial* partials)
{
    __shared__ Partial values[kThreads];
    Partial p{Empty(), Empty()};
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < vocab)
    {
        const bool ts = i >= f.timestamp_first;
        const bool allowed =
            raw || (!(suppressed && suppressed[i]) && ((i < f.eot && f.text) || (i == f.eot && f.end) ||
                                                       (ts && i >= f.timestamp_min && i <= f.timestamp_max)));
        if (allowed)
        {
            const float score = ToFloat(logits[i]);
            if (score != -CUDART_INF_F)
            {
                const Reduction r{score, i, 1.0};
                if (ts && !raw)
                    p.timestamp = r;
                else
                    p.text = r;
            }
        }
    }
    values[threadIdx.x] = p;
    Reduce(values);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = values[0];
}

template <typename T>
__global__ void SelectReduced(const T* logits, const Partial* partials, int blocks, int probability_index,
                              int32_t* token, double* stats)
{
    __shared__ Partial values[kThreads];
    Partial p{Empty(), Empty()};
    for (int i = threadIdx.x; i < blocks; i += kThreads)
    {
        p.text = Merge(p.text, partials[i].text);
        p.timestamp = Merge(p.timestamp, partials[i].timestamp);
    }
    values[threadIdx.x] = p;
    Reduce(values);
    if (threadIdx.x != 0)
        return;
    p = values[0];
    if (probability_index >= 0)
    {
        stats[1] = exp(double(ToFloat(logits[probability_index])) - p.text.maximum) / p.text.sum;
        return;
    }
    const bool force_timestamp =
        p.timestamp.sum > 0 && double(p.timestamp.maximum) + log(p.timestamp.sum) > p.text.maximum;
    const Reduction selected = force_timestamp ? p.timestamp : Merge(p.text, p.timestamp);
    if (!isfinite(selected.maximum) || selected.sum == 0)
    {
        *token = -1;
        return;
    }
    *token = selected.index;
    stats[0] -= log(selected.sum);
}

template <typename T>
void LaunchSelection(cudaStream_t stream, const T* logits, int vocab, const uint8_t* suppressed, TimestampFilter filter,
                     int probability_index, double* workspace, int32_t* token, double* stats)
{
    const int blocks = (vocab + kThreads - 1) / kThreads;
    auto* partials = reinterpret_cast<Partial*>(workspace);
    FilterReduce<<<blocks, kThreads, 0, stream>>>(logits, vocab, suppressed, filter, probability_index >= 0, partials);
    SelectReduced<<<1, kThreads, 0, stream>>>(logits, partials, blocks, probability_index, token, stats);
}

__global__ void SetInputs(int32_t token, bool update_token, int64_t position, int32_t* input, int64_t* index,
                          int64_t* nonpad)
{
    if (update_token)
        *input = token;
    *index = position;
    *nonpad = position + 1;
}

__global__ void HistoryInputs(const int32_t* prompt, int block_size, int count, int64_t position, int32_t* input,
                              int64_t* indices, int64_t* nonpad)
{
    for (int i = threadIdx.x; i < block_size; i += blockDim.x)
    {
        input[i] = i < count ? prompt[position + i] : 0;
        indices[i] = position + i;
    }
    if (threadIdx.x == 0)
        *nonpad = position + count;
}

}  // namespace

void launch_whisper_history_inputs(cudaStream_t stream, const int32_t* prompt, int block_size, int count,
                                   int64_t position, int32_t* input, int64_t* indices, int64_t* nonpad)
{
    HistoryInputs<<<1, kThreads, 0, stream>>>(prompt, block_size, count, position, input, indices, nonpad);
}

void launch_whisper_inputs(cudaStream_t stream, int32_t token, bool update_token, int64_t position, int32_t* input,
                           int64_t* index, int64_t* nonpad)
{
    SetInputs<<<1, 1, 0, stream>>>(token, update_token, position, input, index, nonpad);
}

void launch_whisper_sample(cudaStream_t stream, const void* logits, bool fp16, int vocab, const uint8_t* suppressed,
                           TimestampFilter filter, double* workspace, int32_t* token, double* stats,
                           int probability_index)
{
    if (fp16)
        LaunchSelection(stream, static_cast<const __half*>(logits), vocab, suppressed, filter, probability_index,
                        workspace, token, stats);
    else
        LaunchSelection(stream, static_cast<const float*>(logits), vocab, suppressed, filter, probability_index,
                        workspace, token, stats);
}

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
