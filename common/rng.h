// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace din::common
{
enum class Rng { CpuStd, CudaPhilox, CpuPhilox };

// Reference launch geometry: H100 SXM5, 132 SMs * 2048 threads/SM.
inline constexpr size_t kH100PhiloxStride = 270336;

struct NormalOptions
{
    bool round_to_bfloat16 = false;
    // Normal4 components are separated by this many samples. Must be nonzero.
    size_t philox_stride = kH100PhiloxStride;
};

// Fill a contiguous buffer with N(0, 1) samples, restarting at the seed each call.
// CpuStd preserves std::default_random_engine/normal_distribution behavior,
// including platform dependence and truncation of the seed to unsigned int.
// CpuPhilox uses Philox4x32-10 and cuRAND's Box-Muller convention and ordering.
// CPU math can differ from CUDA, including at BF16 rounding boundaries.
// CudaPhilox must be dispatched through GenerateNormalCuda in rng_cuda.h.
void GenerateNormalCpu(std::span<float> output, uint64_t seed, Rng rng,
                       const NormalOptions& options = {});
}
