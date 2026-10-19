// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

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
void LaunchSeedVrLabKernel(cudaStream_t stream, void *image, const void *reference,
                          size_t height, size_t width, size_t stride);

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void *encoded, void *condition, size_t spatial_size);

void LaunchSeedVrEulerKernel(cudaStream_t stream, const void *latent, const void *velocity, void *sampled,
                             size_t element_count);

void LaunchSeedVrFloatToHalfKernel(cudaStream_t stream, const float *source, void *destination,
                                   size_t element_count);
