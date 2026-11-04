// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "seedvr_cuda_kernels.h"

namespace {
    constexpr int kThreads = 256;

    inline constexpr size_t DivRoundUp(size_t value, size_t divisor) {
        return value / divisor + (value % divisor != 0);
    }

    __device__ float CubicWeight(float x) {
        constexpr float a = -0.75f;
        x = fabsf(x);
        // Keep the reference's individual float operations rather than contracting to FMA.
        if (x <= 1.0f) {
            const float factor = __fsub_rn(__fmul_rn(a + 2.0f, x), a + 3.0f);
            return __fadd_rn(__fmul_rn(__fmul_rn(factor, x), x), 1.0f);
        }
        if (x < 2.0f) {
            const float factor = __fadd_rn(__fmul_rn(__fsub_rn(__fmul_rn(a, x), 5.0f * a), x), 8.0f * a);
            return __fsub_rn(__fmul_rn(factor, x), 4.0f * a);
        }
        return 0.0f;
    }

    __global__ void ResizeBicubic(const float *source, __half *encoder_input, float *reference,
                                  size_t source_height, size_t source_width, size_t target_height, size_t target_width,
                                  size_t padded_height, size_t padded_width, size_t encoder_channel_stride) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t plane = padded_height * padded_width;
        if (i >= 3 * plane) return;
        const size_t channel = i / plane, y = i % plane / padded_width, x = i % padded_width;
        float value = 0.0f;
        if (y < target_height && x < target_width) {
            const float scale_y = __fdiv_rn(static_cast<float>(source_height), static_cast<float>(target_height));
            const float scale_x = __fdiv_rn(static_cast<float>(source_width), static_cast<float>(target_width));
            const float source_y = __fsub_rn(__fmul_rn(__fadd_rn(static_cast<float>(y), 0.5f), scale_y), 0.5f);
            const float source_x = __fsub_rn(__fmul_rn(__fadd_rn(static_cast<float>(x), 0.5f), scale_x), 0.5f);
            const int y_base = static_cast<int>(floorf(source_y)), x_base = static_cast<int>(floorf(source_x));
            float weight_sum = 0.0f;
            for (int ky = -1; ky <= 2; ++ky) {
                const int sy = min(max(y_base + ky, 0), static_cast<int>(source_height) - 1);
                const float wy = CubicWeight(__fsub_rn(source_y, static_cast<float>(y_base + ky)));
                for (int kx = -1; kx <= 2; ++kx) {
                    const int sx = min(max(x_base + kx, 0), static_cast<int>(source_width) - 1);
                    const float weight = __fmul_rn(wy, CubicWeight(__fsub_rn(source_x, static_cast<float>(x_base + kx))));
                    const float pixel = source[(channel * source_height + sy) * source_width + sx];
                    value = __fadd_rn(value, __fmul_rn(pixel, weight));
                    weight_sum = __fadd_rn(weight_sum, weight);
                }
            }
            value = fminf(1.0f, fmaxf(0.0f, __fdiv_rn(value, weight_sum)));
        }
        encoder_input[channel * encoder_channel_stride + i % plane] = __float2half_rn(value);
        reference[i] = value;
    }

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

    __global__ void CropToFloat(const __half *source, float *output, size_t height, size_t width,
                                size_t source_stride, size_t source_channel_stride) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (i >= 3 * plane) return;
        const size_t c = i / plane, y = i % plane / width, x = i % width;
        output[i] = Clamp01(__half2float(source[c * source_channel_stride + y * source_stride + x]));
    }

    __global__ void InitLabLowpass(const __half *generated, const float *reference,
                                   float *generated_low, float *reference_low,
                                   size_t height, size_t width, size_t stride, size_t channel_stride,
                                   size_t reference_channel_stride) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (i >= 3 * plane) return;
        const size_t channel = i / plane;
        const size_t y = i % plane / width, x = i % width;
        const size_t padded = channel * channel_stride + y * stride + x;
        generated_low[i] = Clamp01(__half2float(generated[padded])) * 2.0f - 1.0f;
        reference_low[i] = Clamp01(reference[channel * reference_channel_stride + y * stride + x]) * 2.0f - 1.0f;
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
                               size_t height, size_t width, size_t stride, size_t channel_stride,
                               size_t reference_channel_stride) {
        const size_t p = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t plane = height * width;
        if (p >= plane) return;
        const size_t y = p / width, x = p % width;
        float rgb[3], ref_rgb[3];
        for (size_t c = 0; c < 3; ++c) {
            const size_t padded = c * channel_stride + y * stride + x;
            const float raw = 2.0f * Clamp01(__half2float(generated[padded])) - 1.0f;
            const float combined = fminf(1.0f, fmaxf(-1.0f, raw - generated_low[c * plane + p] +
                                                              reference_low[c * plane + p]));
            rgb[c] = Clamp01((combined + 1.0f) * 0.5f);
            ref_rgb[c] = Clamp01(reference[c * reference_channel_stride + y * stride + x]);
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

    __global__ void FinishLab(const float *lab, __half *image, size_t height, size_t width, size_t stride, size_t channel_stride) {
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
        for (size_t c = 0; c < 3; ++c) image[c * channel_stride + y * stride + x] = __float2half_rn(rgb[c]);
    }

    void CheckCuda(cudaError_t error, const char *operation) {
        if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }

    // Thrust receives a fixed arena instead of cudaMalloc/cudaFree per sort.
    struct SortArena {
        char* base;
        size_t capacity, offset = 0, active = 0;
    };
    struct SortAllocator {
        using value_type = char;
        SortArena* arena;
        char* allocate(std::ptrdiff_t count) {
            const size_t bytes = (static_cast<size_t>(count) + 255) / 256 * 256;
            if (count < 0 || bytes > arena->capacity - arena->offset)
                throw std::runtime_error("SeedVR LAB sort scratch capacity exceeded");
            auto* result = arena->base + arena->offset;
            arena->offset += bytes;
            ++arena->active;
            return result;
        }
        void deallocate(char*, size_t) {
            if (--arena->active == 0) arena->offset = 0;
        }
    };
    template<class T> class DeviceScratch {
    public:
        explicit DeviceScratch(void* data) : data_(static_cast<T*>(data)) {}
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

    __global__ void AugmentCondition(__half *condition, const __half *latent, const float *noise,
                                    float sigma, size_t element_count) {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < element_count) {
            // Upstream FP16 augmentation: 0.1 * base_noise + 0.05 * fresh_noise.
            const __half base = __float2half_rn(__half2float(latent[index]) * 0.1f);
            const __half fresh = __float2half_rn(__half2float(__float2half_rn(noise[index])) * 0.05f);
            const float augmented = __half2float(__hadd(base, fresh));
            condition[index] = __float2half_rn(__fadd_rn(
                __fmul_rn(1.0f - sigma, __half2float(condition[index])), __fmul_rn(sigma, augmented)));
        }
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

void LaunchSeedVrResizeBicubic(cudaStream_t stream, const float *source, void *encoder_input, float *reference,
                               size_t source_height, size_t source_width, size_t target_height, size_t target_width,
                               size_t padded_height, size_t padded_width, size_t encoder_channel_stride) {
    const size_t count = 3 * padded_height * padded_width;
    ResizeBicubic<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        source, static_cast<__half *>(encoder_input), reference, source_height, source_width, target_height,
        target_width, padded_height, padded_width, encoder_channel_stride);
    CheckCuda(cudaGetLastError(), "SeedVR bicubic resize");
}

void LaunchSeedVrGatherTile(cudaStream_t stream, const void *source, void *tile,
                           size_t channels, size_t height, size_t width, SeedVrTile region) {
    const size_t count = channels * region.height * region.width;
    GatherTile<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(source), static_cast<__half *>(tile), channels, height, width, region);
    CheckCuda(cudaGetLastError(), "SeedVR tile gather");
}

void LaunchSeedVrBlendTile(cudaStream_t stream, const void *tile, float *sum, float *weights,
                          size_t channels, size_t height, size_t width, SeedVrTile region) {
    const size_t count = channels * region.height * region.width;
    BlendTile<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(tile), sum, weights, channels, height, width, region);
    CheckCuda(cudaGetLastError(), "SeedVR tile blend");
}

void LaunchSeedVrNormalizeTiles(cudaStream_t stream, const float *sum, const float *weights,
                               void *output, size_t channels, size_t spatial_size) {
    const size_t count = channels * spatial_size;
    NormalizeTiles<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        sum, weights, static_cast<__half *>(output), channels, spatial_size);
    CheckCuda(cudaGetLastError(), "SeedVR tile normalize");
}

void LaunchSeedVrClampKernel(cudaStream_t stream, void *image, size_t element_count) {
    ClampImage<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__half *>(image), element_count);
    CheckCuda(cudaGetLastError(), "SeedVR clamp");
}

void LaunchSeedVrCropToFloat(cudaStream_t stream, const void *source, float *output,
                             size_t height, size_t width, size_t source_stride, size_t source_channel_stride) {
    CropToFloat<<<static_cast<unsigned int>(DivRoundUp(3 * height * width, kThreads)), kThreads, 0, stream>>>(
        static_cast<const __half *>(source), output, height, width, source_stride, source_channel_stride);
    CheckCuda(cudaGetLastError(), "SeedVR output crop and conversion");
}

size_t SeedVrLabScratchBytes(size_t height, size_t width) {
    const size_t plane = height * width;
    if (plane > INT_MAX) throw std::invalid_argument("SeedVR LAB frame exceeds sort index capacity");
    cub::DoubleBuffer<float> keys(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> indices(nullptr, nullptr);
    size_t pairs_bytes = 0, keys_bytes = 0;
    CheckCuda(cub::DeviceRadixSort::SortPairs(nullptr, pairs_bytes, keys, indices, static_cast<std::ptrdiff_t>(plane)), "LAB pair-sort storage query");
    CheckCuda(cub::DeviceRadixSort::SortKeys(nullptr, keys_bytes, keys, static_cast<std::ptrdiff_t>(plane)), "LAB key-sort storage query");
    // Thrust's radix path adds alternate key/value buffers to CUB's scratch.
    const size_t alternate = (plane * sizeof(float) + 127) / 128 * 128;
    const size_t sort_bytes = std::max(pairs_bytes + 2 * alternate, keys_bytes + alternate);
    const size_t image_bytes = (3 * plane * sizeof(float) + 255) / 256 * 256;
    // After PrepareLab the two low-pass buffers are dead. Reuse one for sorted
    // keys/indices and the other for Thrust's temporary storage.
    return 3 * image_bytes + std::max(image_bytes, (sort_bytes + 255) / 256 * 256);
}

void LaunchSeedVrLabKernel(cudaStream_t stream, void *image, const void *reference,
                          size_t height, size_t width, size_t stride, size_t channel_stride,
                          size_t reference_channel_stride,
                          void* scratch, size_t scratch_bytes) {
    const size_t plane = height * width;
    if (plane > UINT32_MAX) throw std::invalid_argument("SeedVR LAB frame exceeds uint32 index capacity");
    const unsigned int pixel_blocks = static_cast<unsigned int>((plane + kThreads - 1) / kThreads);
    const unsigned int channel_blocks = static_cast<unsigned int>((3 * plane + kThreads - 1) / kThreads);
    const size_t image_bytes = (3 * plane * sizeof(float) + 255) / 256 * 256;
    if (!scratch || scratch_bytes < 4 * image_bytes)
        throw std::invalid_argument("SeedVR LAB scratch buffer is too small");
    size_t used = 0;
    auto take = [&](size_t bytes) {
        auto* result = static_cast<char*>(scratch) + used;
        used += (bytes + 255) / 256 * 256;
        return result;
    };
    DeviceScratch<float> generated_low(take(3 * plane * sizeof(float))), reference_low(take(3 * plane * sizeof(float)));
    DeviceScratch<float> generated_next(take(3 * plane * sizeof(float))), reference_next(take(3 * plane * sizeof(float)));
    InitLabLowpass<<<channel_blocks, kThreads, 0, stream>>>(static_cast<const __half *>(image),
        static_cast<const float *>(reference), generated_low.get(), reference_low.get(), height, width, stride,
        channel_stride, reference_channel_stride);
    CheckCuda(cudaGetLastError(), "SeedVR LAB initialization");
    float *generated_current = generated_low.get(), *reference_current = reference_low.get();
    float *generated_temp = generated_next.get(), *reference_temp = reference_next.get();
    for (int level = 0; level < 5; ++level) {
        const int radius = std::max(1, std::min<int>(1 << level, static_cast<int>(std::min(height, width) / 8)));
        LabWaveletBlur<<<channel_blocks, kThreads, 0, stream>>>(generated_current, generated_temp, height, width, radius);
        CheckCuda(cudaGetLastError(), "SeedVR LAB generated blur");
        LabWaveletBlur<<<channel_blocks, kThreads, 0, stream>>>(reference_current, reference_temp, height, width, radius);
        CheckCuda(cudaGetLastError(), "SeedVR LAB reference blur");
        std::swap(generated_current, generated_temp);
        std::swap(reference_current, reference_temp);
    }
    // The unused ping-pong buffers now hold LAB channels. The five wavelet high-pass
    // contributions telescope to generated_raw - generated_low.
    float *generated_lab = generated_temp, *reference_lab = reference_temp;
    PrepareLab<<<pixel_blocks, kThreads, 0, stream>>>(static_cast<const __half *>(image),
        static_cast<const float *>(reference), generated_current, reference_current,
        generated_lab, reference_lab, height, width, stride, channel_stride, reference_channel_stride);
    CheckCuda(cudaGetLastError(), "SeedVR LAB preparation");
    DeviceScratch<float> generated_sorted(generated_current), reference_sorted(generated_current + plane);
    DeviceScratch<uint32_t> indices(generated_current + 2 * plane);
    // Five blur levels leave reference_current in the final allocated image
    // block, followed by any extra queried sort storage.
    auto* sort_base = reinterpret_cast<char*>(reference_current);
    SortArena arena{sort_base, scratch_bytes - static_cast<size_t>(sort_base - static_cast<char*>(scratch))};
    SortAllocator allocator{&arena};
    const auto policy = thrust::cuda::par_nosync(allocator).on(stream);
    for (int channel = 0; channel < 3; ++channel) {
        const size_t offset = static_cast<size_t>(channel) * plane;
        PrepareLabSort<<<pixel_blocks, kThreads, 0, stream>>>(generated_lab + offset, reference_lab + offset,
            generated_sorted.get(), reference_sorted.get(), indices.get(), plane);
        CheckCuda(cudaGetLastError(), "SeedVR LAB sort preparation");
        thrust::sort_by_key(policy, thrust::device_pointer_cast(generated_sorted.get()),
            thrust::device_pointer_cast(generated_sorted.get() + plane), thrust::device_pointer_cast(indices.get()));
        thrust::sort(policy, thrust::device_pointer_cast(reference_sorted.get()),
            thrust::device_pointer_cast(reference_sorted.get() + plane));
        ScatterLabMatch<<<pixel_blocks, kThreads, 0, stream>>>(reference_sorted.get(),
            indices.get(), generated_lab, plane, channel);
        CheckCuda(cudaGetLastError(), "SeedVR LAB scatter");
    }
    FinishLab<<<pixel_blocks, kThreads, 0, stream>>>(generated_lab, static_cast<__half *>(image), height, width, stride, channel_stride);
    CheckCuda(cudaGetLastError(), "SeedVR LAB kernels");
}

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void *encoded, void *condition, size_t spatial_size) {
    const size_t count = 17 * spatial_size;
    PackCondition<<<static_cast<unsigned int>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(encoded), static_cast<__half *>(condition), spatial_size);
    CheckCuda(cudaGetLastError(), "SeedVR conditioning");
}

void LaunchSeedVrConditionNoiseKernel(cudaStream_t stream, void *condition, const void *latent,
                                     const float *noise, float sigma, size_t element_count) {
    AugmentCondition<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__half *>(condition), static_cast<const __half *>(latent), noise, sigma, element_count);
    CheckCuda(cudaGetLastError(), "SeedVR conditioning noise");
}

void LaunchSeedVrEulerKernel(cudaStream_t stream, const void *latent, const void *velocity, void *sampled,
                             size_t element_count) {
    EulerOneStep<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __half *>(latent), static_cast<const __half *>(velocity), static_cast<__half *>(sampled),
        element_count);
    CheckCuda(cudaGetLastError(), "SeedVR Euler step");
}

void LaunchSeedVrFloatToHalfKernel(cudaStream_t stream, const float *source, void *destination,
                                   size_t element_count) {
    FloatToHalf<<<static_cast<unsigned int>((element_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        source, static_cast<__half *>(destination), element_count);
    CheckCuda(cudaGetLastError(), "SeedVR float to half");
}
