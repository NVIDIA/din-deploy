// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam2_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace din::sam2 {
namespace {
void Check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string("SAM2 CUDA: ")+cudaGetErrorString(status));
}
__global__ void Convert(DeviceRgbaView src, uint8_t* rgb) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=src.width || y>=src.height) return;
    const auto* pixel=reinterpret_cast<const float*>(static_cast<const char*>(src.data)+y*src.row_bytes)+4*x;
    const auto offset=(size_t(src.height-1-y)*src.width+x)*3;
    for(int c=0;c<3;++c) {
        float v=pixel[c];
        if(src.premultiplied) v=pixel[3]>0 ? v/pixel[3] : 0.f;
        v=isfinite(v) ? fminf(fmaxf(v,0.f),1.f) : 0.f;
        rgb[offset+c]=static_cast<uint8_t>(roundf(v*255.f));
    }
}
__global__ void Matte(DeviceRgbaView src, DeviceOutputView dst, const __half* logits, int size) {
    const int dx=blockIdx.x*blockDim.x+threadIdx.x, dy=blockIdx.y*blockDim.y+threadIdx.y;
    if(dx>=dst.width || dy>=dst.height) return;
    const int x=dst.x+dx, y=dst.y+dy;
    auto* out=reinterpret_cast<float*>(static_cast<char*>(dst.data)+dy*dst.row_bytes)+4*dx;
    if(x<0 || y<0 || x>=src.width || y>=src.height) {
        for(int c=0;c<4;++c) out[c]=0; return;
    }
    // ONNX Resize: linear, half_pixel, edge replication, followed by logits > 0.
    const float fx=fmaxf(0.f,(x+0.5f)*size/src.width-0.5f);
    const float fy=fmaxf(0.f,(src.height-1-y+0.5f)*size/src.height-0.5f);
    const int x0=min(int(fx),size-1), y0=min(int(fy),size-1);
    const int x1=min(x0+1,size-1), y1=min(y0+1,size-1);
    const float ax=fx-x0, ay=fy-y0;
    const float a=__half2float(logits[y0*size+x0])*(1-ax)+__half2float(logits[y0*size+x1])*ax;
    const float b=__half2float(logits[y1*size+x0])*(1-ax)+__half2float(logits[y1*size+x1])*ax;
    const float m=(a*(1-ay)+b*ay)>0 ? 1.f : 0.f;
    if(dst.masked_alpha) {
        const auto* in=reinterpret_cast<const float*>(static_cast<const char*>(src.data)+y*src.row_bytes)+4*x;
        for(int c=0;c<3;++c) out[c]=in[c]*(src.premultiplied?m:1.f);
        out[3]=in[3]*m;
    } else {out[0]=out[1]=out[2]=m; out[3]=1.f;}
}
}
void ConvertDeviceRgba(const DeviceRgbaView& src, uint8_t* rgb, void* stream) {
    Convert<<<dim3((src.width+15)/16,(src.height+15)/16),dim3(16,16),0,static_cast<cudaStream_t>(stream)>>>(src,rgb);
    Check(cudaGetLastError());
}
void WriteDeviceMatte(const DeviceRgbaView& src, const DeviceOutputView& dst, const void* logits, int size, void* stream) {
    if(dst.width<=0 || dst.height<=0) return;
    Matte<<<dim3((dst.width+15)/16,(dst.height+15)/16),dim3(16,16),0,static_cast<cudaStream_t>(stream)>>>(src,dst,static_cast<const __half*>(logits),size);
    Check(cudaGetLastError());
}
}
