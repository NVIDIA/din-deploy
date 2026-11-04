// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// Source and reference are CHW; encoder channels may span multiple video frames.
void LaunchSeedVrResizeBicubic(cudaStream_t stream, const float *source, void *encoder_input, float *reference,
                               size_t source_height, size_t source_width, size_t target_height, size_t target_width,
                               size_t padded_height, size_t padded_width, size_t encoder_channel_stride);

// Coordinates and fades are in the output canvas's spatial units.
struct SeedVrTile {
    size_t y, x, height, width;
    size_t top, bottom, left, right;
};

void LaunchSeedVrGatherTile(cudaStream_t stream, const void *source, void *tile,
                           size_t channels, size_t height, size_t width, SeedVrTile region);
void LaunchSeedVrBlendTile(cudaStream_t stream, const void *tile, float *sum, float *weights,
                          size_t channels, size_t height, size_t width, SeedVrTile region);
void LaunchSeedVrNormalizeTiles(cudaStream_t stream, const float *sum, const float *weights,
                               void *output, size_t channels, size_t spatial_size);
void LaunchSeedVrClampKernel(cudaStream_t stream, void *image, size_t element_count);
// Includes image scratch and bounded temporary storage for Thrust sorting.
size_t SeedVrLabScratchBytes(size_t height, size_t width);
void LaunchSeedVrLabKernel(cudaStream_t stream, void *image, const void *reference,
                          size_t height, size_t width, size_t stride, size_t channel_stride,
                          size_t reference_channel_stride,
                          void *scratch, size_t scratch_bytes);

void LaunchSeedVrCropToFloat(cudaStream_t stream, const void *source, float *output,
                             size_t height, size_t width, size_t source_stride, size_t source_channel_stride);

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void *encoded, void *condition, size_t spatial_size);

void LaunchSeedVrConditionNoiseKernel(cudaStream_t stream, void *condition, const void *latent,
                                     const float *noise, float sigma, size_t element_count);

void LaunchSeedVrEulerKernel(cudaStream_t stream, const void *latent, const void *velocity, void *sampled,
                             size_t element_count);

void LaunchSeedVrFloatToHalfKernel(cudaStream_t stream, const float *source, void *destination,
                                   size_t element_count);
