// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "seedvr_cuda_kernels.h"

namespace {
    constexpr int kThreads = 256;

    __global__ void GatherTile(const __half *source, __half *tile, size_t channels,
                               size_t height, size_t width, SeedVrTile r) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = r.height * r.width;
        if (i >= channels * plane) return;
        const size_t y = i % plane / r.width, x = i % r.width;
        tile[i] = source[(i / plane * height + r.y + y) * width + r.x + x];
    }

    __device__ float Fade(size_t position, size_t overlap) {
        // Half-sample centers keep weights positive, including a one-pixel overlap.
        return 0.5f - 0.5f * cosf(3.14159265358979323846f *
                                (static_cast<float>(position) + 0.5f) / static_cast<float>(overlap));
    }

    __global__ void BlendTile(const __half *tile, float *sum, float *weights, size_t channels,
                              size_t height, size_t width, SeedVrTile r) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = r.height * r.width;
        if (i >= channels * plane) return;
        const size_t c = i / plane, y = i % plane / r.width, x = i % r.width;
        float weight = 1.0f;
        if (y < r.top) weight *= Fade(y, r.top);
        if (y >= r.height - r.bottom) weight *= Fade(r.height - 1 - y, r.bottom);
        if (x < r.left) weight *= Fade(x, r.left);
        if (x >= r.width - r.right) weight *= Fade(r.width - 1 - x, r.right);
        const size_t p = (r.y + y) * width + r.x + x;
        // Tiles execute sequentially on one stream; each thread owns one destination.
        sum[c * height * width + p] += __half2float(tile[i]) * weight;
        if (c == 0) weights[p] += weight;
    }

    __global__ void NormalizeTiles(const float *sum, const float *weights, __half *output,
                                   size_t channels, size_t spatial_size) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= channels * spatial_size) return;
        float value = sum[i] / weights[i % spatial_size];
        output[i] = __float2half_rn(value);
    }

    __global__ void ClampImage(__half *image, size_t element_count) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < element_count) {
            image[i] = __float2half_rn(fminf(1.0f, fmaxf(0.0f, __half2float(image[i]))));
        }
    }

    constexpr float kWhiteX = 0.95047f;
    constexpr float kWhiteZ = 1.08883f;
    constexpr float kDelta = 6.0f / 29.0f;
    constexpr float kKappa = (29.0f / 3.0f) * (29.0f / 3.0f) * (29.0f / 3.0f);

    __device__ float Clamp01(float value) { return fminf(1.0f, fmaxf(0.0f, value)); }

    __global__ void InitLabLowpass(const __half *generated, const float *reference,
                                   float *generated_low, float *reference_low,
                                   size_t height, size_t width, size_t stride) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (i >= 3 * plane) return;
        const size_t channel = i / plane;
        const size_t y = i % plane / width, x = i % width;
        const size_t padded = channel * height * stride + y * stride + x;
        generated_low[i] = Clamp01(__half2float(generated[padded])) * 2.0f - 1.0f;
        reference_low[i] = Clamp01(reference[padded]) * 2.0f - 1.0f;
    }

    __global__ void LabWaveletBlur(const float *source, float *target, size_t height, size_t width, int radius) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (i >= 3 * plane) return;
        const size_t channel = i / plane, y = i % plane / width, x = i % width;
        const size_t offset = channel * plane;
        constexpr float weights[3] = {0.25f, 0.5f, 0.25f};
        float value = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
            const size_t sy = min(height - 1, static_cast<size_t>(max(0, static_cast<int>(y) + ky * radius)));
            for (int kx = -1; kx <= 1; ++kx) {
                const size_t sx = min(width - 1, static_cast<size_t>(max(0, static_cast<int>(x) + kx * radius)));
                value += weights[ky + 1] * weights[kx + 1] * source[offset + sy * width + sx];
            }
        }
        target[i] = value;
    }

    __device__ float SrgbToLinear(float x) {
        return x > 0.04045f ? powf((x + 0.055f) / 1.055f, 2.4f) : x / 12.92f;
    }

    __device__ float LabCurve(float x) {
        return x > kDelta * kDelta * kDelta ? cbrtf(x) : (x * kKappa + 16.0f) / 116.0f;
    }

    __device__ void RgbToLab(float red, float green, float blue, float &lightness, float &a, float &b) {
        red = SrgbToLinear(Clamp01(red));
        green = SrgbToLinear(Clamp01(green));
        blue = SrgbToLinear(Clamp01(blue));
        const float fx = LabCurve((0.4124564f * red + 0.3575761f * green + 0.1804375f * blue) / kWhiteX);
        const float fy = LabCurve(0.2126729f * red + 0.7151522f * green + 0.0721750f * blue);
        const float fz = LabCurve((0.0193339f * red + 0.1191920f * green + 0.9503041f * blue) / kWhiteZ);
        lightness = 116.0f * fy - 16.0f;
        a = 500.0f * (fx - fy);
        b = 200.0f * (fy - fz);
    }

    __global__ void PrepareLab(const __half *generated, const float *reference,
                               const float *generated_low, const float *reference_low,
                               float *generated_lab, float *reference_lab,
                               size_t height, size_t width, size_t stride) {
        const size_t p = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (p >= plane) return;
        const size_t y = p / width, x = p % width;
        float rgb[3], ref_rgb[3];
        for (size_t c = 0; c < 3; ++c) {
            const size_t padded = c * height * stride + y * stride + x;
            const float raw = 2.0f * Clamp01(__half2float(generated[padded])) - 1.0f;
            const float combined = fminf(1.0f, fmaxf(-1.0f, raw - generated_low[c * plane + p] +
                                                              reference_low[c * plane + p]));
            rgb[c] = Clamp01((combined + 1.0f) * 0.5f);
            ref_rgb[c] = Clamp01(reference[padded]);
        }
        RgbToLab(rgb[0], rgb[1], rgb[2], generated_lab[p], generated_lab[plane + p],
                 generated_lab[2 * plane + p]);
        RgbToLab(ref_rgb[0], ref_rgb[1], ref_rgb[2], reference_lab[p], reference_lab[plane + p],
                 reference_lab[2 * plane + p]);
    }

    __global__ void PrepareLabSort(const float *generated, const float *reference,
                                   float *generated_sorted, float *reference_sorted,
                                   uint32_t *indices, size_t count) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        generated_sorted[i] = generated[i];
        reference_sorted[i] = reference[i];
        indices[i] = static_cast<uint32_t>(i);
    }

    __global__ void ScatterLabMatch(const float *reference_sorted, const uint32_t *indices, float *generated_lab,
                                    size_t count, int channel) {
        const size_t rank = blockIdx.x * blockDim.x + threadIdx.x;
        if (rank >= count) return;
        const size_t original = indices[rank];
        const size_t offset = static_cast<size_t>(channel) * count + original;
        // ComfyUI preserves 80% of generated L and matches all a/b chroma.
        generated_lab[offset] = channel == 0
            ? 0.8f * generated_lab[offset] + 0.2f * reference_sorted[rank]
            : reference_sorted[rank];
    }

    __device__ float InverseLabCurve(float x) {
        return x > kDelta ? x * x * x : (116.0f * x - 16.0f) / kKappa;
    }

    __device__ float LinearToSrgb(float x) {
        return Clamp01(x > 0.0031308f ? 1.055f * powf(fmaxf(x, 0.0f), 1.0f / 2.4f) - 0.055f
                                       : 12.92f * x);
    }

    __global__ void FinishLab(const float *lab, __half *image, size_t height, size_t width, size_t stride) {
        const size_t p = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (p >= plane) return;
        const float fy = (lab[p] + 16.0f) / 116.0f;
        const float fx = fy + lab[plane + p] / 500.0f;
        const float fz = fy - lab[2 * plane + p] / 200.0f;
        const float X = kWhiteX * InverseLabCurve(fx);
        const float Y = InverseLabCurve(fy);
        const float Z = kWhiteZ * InverseLabCurve(fz);
        const float rgb[3] = {
            LinearToSrgb(3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z),
            LinearToSrgb(-0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z),
            LinearToSrgb(0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z)
        };
        const size_t y = p / width, x = p % width;
        for (size_t c = 0; c < 3; ++c) image[c * height * stride + y * stride + x] = __float2half_rn(rgb[c]);
    }

    void CheckCuda(cudaError_t error, const char *operation) {
        if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }

    template<class T> class DeviceScratch {
    public:
        explicit DeviceScratch(size_t count) {
            CheckCuda(cudaMalloc(&data_, count * sizeof(T)), "SeedVR LAB cudaMalloc");
        }
        ~DeviceScratch() { cudaFree(data_); }
        DeviceScratch(const DeviceScratch &) = delete;
        DeviceScratch &operator=(const DeviceScratch &) = delete;
        T *get() const { return data_; }
    private:
        T *data_ = nullptr;
    };

    __global__ void PackCondition(const __half *encoded, __half *condition, size_t spatial_size) {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t element_count = 17 * spatial_size;
        if (index >= element_count) {
            return;
        }
        condition[index] = index < 16 * spatial_size ? encoded[index] : __float2half(1.0f);
    }

    __global__ void EulerOneStep(const __half *latent, const __half *velocity, __half *sampled, size_t element_count) {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < element_count) {
            sampled[index] = __hsub(latent[index], velocity[index]);
        }
    }

    __global__ void FloatToHalf(const float *source, __half *destination, size_t element_count) {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < element_count) {
            destination[index] = __float2half_rn(source[index]);
        }
    }
} // namespace

void LaunchSeedVrGatherTile(cudaStream_t stream, const void *source, void *tile,
                           size_t channels, size_t height, size_t width, SeedVrTile region) {
    const size_t count = channels * region.height * region.width;
    GatherTile<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(source), static_cast<__half *>(tile), channels, height, width, region);
}

void LaunchSeedVrBlendTile(cudaStream_t stream, const void *tile, float *sum, float *weights,
                          size_t channels, size_t height, size_t width, SeedVrTile region) {
    const size_t count = channels * region.height * region.width;
    BlendTile<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(tile), sum, weights, channels, height, width, region);
}

void LaunchSeedVrNormalizeTiles(cudaStream_t stream, const float *sum, const float *weights,
                               void *output, size_t channels, size_t spatial_size) {
    const size_t count = channels * spatial_size;
    NormalizeTiles<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        sum, weights, static_cast<__half *>(output), channels, spatial_size);
}

void LaunchSeedVrClampKernel(cudaStream_t stream, void *image, size_t element_count) {
    ClampImage<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__half *>(image), element_count);
}

void LaunchSeedVrLabKernel(cudaStream_t stream, void *image, const void *reference,
                          size_t height, size_t width, size_t stride) {
    const size_t plane = height * width;
    if (plane > UINT32_MAX) throw std::invalid_argument("SeedVR LAB frame exceeds uint32 index capacity");
    const unsigned int pixel_blocks = static_cast<unsigned int>((plane + kThreads - 1) / kThreads);
    const unsigned int channel_blocks = static_cast<unsigned int>((3 * plane + kThreads - 1) / kThreads);
    DeviceScratch<float> generated_low(3 * plane), reference_low(3 * plane);
    DeviceScratch<float> generated_next(3 * plane), reference_next(3 * plane);
    InitLabLowpass<<<channel_blocks, kThreads, 0, stream>>>(static_cast<const __half *>(image),
        static_cast<const float *>(reference), generated_low.get(), reference_low.get(), height, width, stride);
    float *generated_current = generated_low.get(), *reference_current = reference_low.get();
    float *generated_temp = generated_next.get(), *reference_temp = reference_next.get();
    for (int level = 0; level < 5; ++level) {
        const int radius = std::max(1, std::min<int>(1 << level, static_cast<int>(std::min(height, width) / 8)));
        LabWaveletBlur<<<channel_blocks, kThreads, 0, stream>>>(generated_current, generated_temp, height, width, radius);
        LabWaveletBlur<<<channel_blocks, kThreads, 0, stream>>>(reference_current, reference_temp, height, width, radius);
        std::swap(generated_current, generated_temp);
        std::swap(reference_current, reference_temp);
    }
    // The unused ping-pong buffers now hold LAB channels. The five wavelet high-pass
    // contributions telescope to generated_raw - generated_low.
    float *generated_lab = generated_temp, *reference_lab = reference_temp;
    PrepareLab<<<pixel_blocks, kThreads, 0, stream>>>(static_cast<const __half *>(image),
        static_cast<const float *>(reference), generated_current, reference_current,
        generated_lab, reference_lab, height, width, stride);
    DeviceScratch<float> generated_sorted(plane), reference_sorted(plane);
    DeviceScratch<uint32_t> indices(plane);
    const auto policy = thrust::cuda::par.on(stream);
    for (int channel = 0; channel < 3; ++channel) {
        const size_t offset = static_cast<size_t>(channel) * plane;
        PrepareLabSort<<<pixel_blocks, kThreads, 0, stream>>>(generated_lab + offset, reference_lab + offset,
            generated_sorted.get(), reference_sorted.get(), indices.get(), plane);
        thrust::sort_by_key(policy, thrust::device_pointer_cast(generated_sorted.get()),
            thrust::device_pointer_cast(generated_sorted.get() + plane), thrust::device_pointer_cast(indices.get()));
        thrust::sort(policy, thrust::device_pointer_cast(reference_sorted.get()),
            thrust::device_pointer_cast(reference_sorted.get() + plane));
        ScatterLabMatch<<<pixel_blocks, kThreads, 0, stream>>>(reference_sorted.get(),
            indices.get(), generated_lab, plane, channel);
    }
    FinishLab<<<pixel_blocks, kThreads, 0, stream>>>(generated_lab, static_cast<__half *>(image), height, width, stride);
    CheckCuda(cudaGetLastError(), "SeedVR LAB kernels");
}

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void *encoded, void *condition, size_t spatial_size) {
    const size_t count = 17 * spatial_size;
    PackCondition<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(encoded), static_cast<__half *>(condition), spatial_size);
}

void LaunchSeedVrEulerKernel(cudaStream_t stream, const void *latent, const void *velocity, void *sampled,
                             size_t element_count) {
    EulerOneStep<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(latent), static_cast<const __half *>(velocity), static_cast<__half *>(sampled),
        element_count);
}

void LaunchSeedVrFloatToHalfKernel(cudaStream_t stream, const float *source, void *destination,
                                   size_t element_count) {
    FloatToHalf<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        source, static_cast<__half *>(destination), element_count);
}
