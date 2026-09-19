// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

void LaunchSeedVrConditionKernel(cudaStream_t stream, const void* encoded, void* condition, size_t spatial_size);
void LaunchSeedVrEulerKernel(cudaStream_t stream, const void* latent, const void* velocity, void* sampled,
                             size_t element_count);
