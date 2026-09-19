// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(DIN_WHISPER_CUDA) || defined(__CUDACC__)
#include <cuda_runtime_api.h>
#endif

#include <cstdint>

namespace din::asr::whisper
{
struct TimestampFilter
{
    int eot, timestamp_first, timestamp_min, timestamp_max;
    bool text = true, end = true;  // An empty timestamp range disables timestamps.
};

#if defined(DIN_WHISPER_CUDA) || defined(__CUDACC__)
// Greedy argmax over logits[lower, upper) of a single-position logits row and
// writes the winning vocab index to *out_index (device memory). `is_fp16`
// selects the logits element type (FLOAT16 vs FLOAT). Runs on `stream`.
void launch_whisper_argmax(cudaStream_t stream, const void* logits, bool is_fp16, int64_t lower, int64_t upper,
                           int32_t* out_index);

// Workspace: four doubles per 256 vocabulary entries.
// Calls sharing a workspace must execute serially on the same stream.
inline int64_t whisper_sampling_workspace_size(int vocab)
{
    return 4 * ((static_cast<int64_t>(vocab) + 255) / 256);
}
void launch_whisper_history_inputs(cudaStream_t stream, const int32_t* prompt, int block_size, int count,
                                   int64_t position, int32_t* input, int64_t* indices, int64_t* nonpad);
void launch_whisper_inputs(cudaStream_t stream, int32_t token, bool update_token, int64_t position, int32_t* input,
                           int64_t* index, int64_t* nonpad);

// Select a timestamp-filtered token and accumulate log probability in stats[0].
// With probability_index >= 0, write its raw probability to stats[1] instead;
// suppression/filter are ignored and token may be null.
void launch_whisper_sample(cudaStream_t stream, const void* logits, bool fp16, int vocab,
                           const uint8_t* suppressed, TimestampFilter filter, double* workspace,
                           int32_t* token, double* stats, int probability_index = -1);

#endif

}  // namespace din::asr::whisper
