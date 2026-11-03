// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rng.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include <Random123/philox.h>

namespace din::common
{
namespace
{
float round_bfloat16(float value)
{
    uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        bits |= 0x00400000U;  // Preserve NaNs as quiet NaNs.
    else
        bits += 0x7fffU + ((bits >> 16) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

void box_muller(uint32_t x, uint32_t y, float* result)
{
    constexpr float a = 2.3283064e-10f;
    constexpr float b = a * 6.2831855f;
    // Match CUDA's fused multiply/add, including integer-to-FP32 conversion.
    const float u = std::fma(static_cast<float>(x), a, a / 2);
    const float v = std::fma(static_cast<float>(y), b, b / 2);
    const float radius = std::sqrt(-2.0f * std::log(u));
    result[0] = radius * std::sin(v);
    result[1] = radius * std::cos(v);
}
}  // namespace

void GenerateNormalCpu(std::span<float> output, uint64_t seed, Rng rng, const NormalOptions& options)
{
    if (rng != Rng::CpuStd && rng != Rng::CpuPhilox)
        throw std::invalid_argument("CPU normal generation requires CpuStd or CpuPhilox");
    if (options.philox_stride == 0 || options.philox_stride > std::numeric_limits<size_t>::max() / 4)
        throw std::invalid_argument("Philox stride must be positive and allow four components without overflow");
    const size_t count = output.size();
    if (rng == Rng::CpuStd)
    {
        std::default_random_engine engine(static_cast<unsigned int>(seed));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (size_t i = 0; i < count; ++i)
        {
            const float sample = normal(engine);
            output[i] = options.round_to_bfloat16 ? round_bfloat16(sample) : sample;
        }
        return;
    }

    using Philox = r123::Philox4x32_R<10>;
    const Philox::key_type key = {{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)}};
    const size_t step = 4 * options.philox_stride;
    for (size_t thread = 0; thread < std::min(count, options.philox_stride); ++thread)
    {
        uint64_t iteration = 0;
        for (size_t base = thread; base < count; ++iteration)
        {
            const Philox::ctr_type counter = {{static_cast<uint32_t>(iteration), static_cast<uint32_t>(iteration >> 32),
                                               static_cast<uint32_t>(thread),
                                               static_cast<uint32_t>(static_cast<uint64_t>(thread) >> 32)}};
            const auto words = Philox{}(counter, key);
            float samples[4];
            box_muller(words[0], words[1], samples);
            box_muller(words[2], words[3], samples + 2);
            for (size_t component = 0; component < 4; ++component)
            {
                const size_t delta = component * options.philox_stride;
                if (delta >= count - base)
                    break;
                const size_t index = base + delta;
                output[index] = options.round_to_bfloat16 ? round_bfloat16(samples[component]) : samples[component];
            }
            if (count - base <= step)
                break;
            base += step;
        }
    }
}
}  // namespace din::common
