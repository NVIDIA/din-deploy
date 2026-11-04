// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../common/device_frame.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <stdexcept>
namespace din::sam::detail {
inline void ValidateRgba(const vision::DeviceRgbaView& source,const vision::DeviceOutputView* output,int device) {
    auto check=[](cudaError_t status) {if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));};
    int current=0;check(cudaGetDevice(&current));
    if(current!=device) throw std::invalid_argument("SAM CUDA device changed");
    auto pointer=[&](const void* data) {
        cudaPointerAttributes attributes{};check(cudaPointerGetAttributes(&attributes,data));
        if(attributes.type!=cudaMemoryTypeDevice || attributes.device!=device) throw std::invalid_argument("SAM borrowed image is on a different CUDA device");
    };
    if(!source.data || source.width<=0 || source.height<=0 || source.width>8192 || source.height>8192 ||
       source.row_bytes==PTRDIFF_MIN || std::abs(source.row_bytes)<ptrdiff_t(source.width)*16 || source.row_bytes%4)
        throw std::invalid_argument("Invalid SAM RGBA source");
    pointer(source.data);
    if(output) {
        if(!output->data || output->width<0 || output->height<0 || output->row_bytes==PTRDIFF_MIN ||
           std::abs(output->row_bytes)<ptrdiff_t(output->width)*16 || output->row_bytes%4)
            throw std::invalid_argument("Invalid SAM RGBA output");
        pointer(output->data);
    }
}
}
