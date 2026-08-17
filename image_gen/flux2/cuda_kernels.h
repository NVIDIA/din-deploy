// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

void launch_flux_euler_kernel(cudaStream_t stream, float t_curr, float t_next, size_t total_elements,
                              float* hidden_states, const float* transformer_output);

void launch_flux_postprocess_kernel(cudaStream_t stream, const float* input_data, float* output_data,
                                    const float* bn_mean, const float* bn_std, unsigned int channels,
                                    unsigned int latent_height, unsigned int latent_width, unsigned int patch_i,
                                    unsigned int patch_j, size_t total_elements);
