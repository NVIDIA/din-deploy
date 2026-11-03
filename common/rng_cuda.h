// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace din::common
{
// Asynchronously fill a contiguous device buffer with N(0, 1) samples.
// Fresh Philox seed, offset zero. Uses ATen's device-dependent launch geometry.
// An empty buffer is a no-op. The buffer and stream must remain valid until done.
void GenerateNormalCuda(cudaStream_t stream, float* output, size_t count, uint64_t seed,
                        bool round_to_bfloat16 = false);

// Generate the same [batch, channels, spatial] samples directly into contiguous
// [batch, spatial, channels] storage. count must be a multiple of channels * spatial.
void GenerateNormalCudaPacked(cudaStream_t stream, float* output, size_t count, uint64_t seed, size_t channels,
                              size_t spatial, bool round_to_bfloat16 = false);
}  // namespace din::common
