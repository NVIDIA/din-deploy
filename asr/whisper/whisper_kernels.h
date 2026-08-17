// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace din::asr::whisper
{

// Greedy argmax over logits[lower, upper) of a single-position logits row and
// writes the winning vocab index to *out_index (device memory). `is_fp16`
// selects the logits element type (FLOAT16 vs FLOAT). Runs on `stream`.
void launch_whisper_argmax(cudaStream_t stream, const void* logits, bool is_fp16, int64_t lower, int64_t upper,
                           int32_t* out_index);

}  // namespace din::asr::whisper
