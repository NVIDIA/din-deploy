// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_cuda.h"
#include "sam3_nvtx.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace din::sam3 {
namespace {
constexpr int Size = 1008;
__device__ float Probability(float x) { return x>=0 ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x)); }
__global__ void Select(const float* classes, const float* presence, int queries, float threshold,
                       int* selected, float* valid, int* error) {
    int count=0;
    for(int i=0;i<16;++i) {selected[i]=-1; valid[i]=0;}
    if(!isfinite(presence[0])) { *error=2; return; }
    for(int i=0;i<queries;++i) {
        if(!isfinite(classes[i])) {*error=2; continue;}
        if(Probability(classes[i])*Probability(presence[0])<=threshold) continue;
        if(count<16) {selected[count]=i; valid[count]=1;}
        ++count;
    }
    if(count>16) { *error=1; for(int i=0;i<16;++i) {selected[i]=-1;valid[i]=0;} }
}
__global__ void Gather(const float* masks,const int* selected,float* packed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16*288*288) return;
    int slot=i/(288*288), query=selected[slot];
    packed[i]=query<0 ? -1000.f : masks[query*288*288+i%(288*288)];
}
__global__ void ClearPointers(float* pointers,const float* valid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<16*256 && valid[i/256]==0) pointers[i]=0;
}
__global__ void Position(float distance,const float* weights,const float* bias,float* output) {
    int c=threadIdx.x;
    float value=bias?bias[c]:0.f;
    for(int j=0;j<(weights?256:1);++j) {
        int k=weights?j:c;
        float phase=distance/powf(10000.f,2.f*((k%128)/2)/128.f);
        float sine=k<128?sinf(phase):cosf(phase);
        value+=weights?weights[c*256+j]*sine:sine;
    }
    for(int slot=0;slot<16;++slot) output[slot*256+c]=value;
}
__global__ void RgbaMasks(vision::DeviceRgbaView src,vision::DeviceOutputView dst,
                          const float* masks,const float* valid,const float* logits,int selected,bool labels) {
    int ox=blockIdx.x*blockDim.x+threadIdx.x, oy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ox>=dst.width || oy>=dst.height) return;
    int x=dst.x+ox,y=dst.y+oy;
    auto* out=reinterpret_cast<float*>(static_cast<char*>(dst.data)+oy*dst.row_bytes)+4*ox;
    if(x<0 || y<0 || x>=src.width || y>=src.height) {for(int c=0;c<4;++c) out[c]=0;return;}
    float fx=fmaxf(0.f,(x+.5f)*288/src.width-.5f), fy=fmaxf(0.f,(src.height-1-y+.5f)*288/src.height-.5f);
    int x0=min(int(fx),287),x1=min(x0+1,287),y0=min(int(fy),287),y1=min(y0+1,287);
    float best=0; int id=0;
    for(int slot=0;slot<16;++slot) {
        if(valid[slot]==0 || (logits && !(logits[slot]>0)) || (selected && slot+1!=selected)) continue;
        const float* mask=masks+slot*288*288;
        float a=mask[y0*288+x0]*(1-(fx-x0))+mask[y0*288+x1]*(fx-x0);
        float b=mask[y1*288+x0]*(1-(fx-x0))+mask[y1*288+x1]*(fx-x0);
        float value=a*(1-(fy-y0))+b*(fy-y0);
        if(value>best) {best=value;id=slot+1;}
    }
    float matte=id?1.f:0.f;
    if(labels) {
        unsigned hash=unsigned(id)*2654435761u;
        out[0]=id?(.25f+.75f*float(hash&255)/255):0;
        out[1]=id?(.25f+.75f*float((hash>>8)&255)/255):0;
        out[2]=id?(.25f+.75f*float((hash>>16)&255)/255):0;out[3]=1;
    } else if(dst.masked_alpha) {
        const auto* in=reinterpret_cast<const float*>(static_cast<const char*>(src.data)+y*src.row_bytes)+4*x;
        for(int c=0;c<3;++c) out[c]=in[c]*(src.premultiplied?matte:1.f);
        out[3]=in[3]*matte;
    } else {out[0]=out[1]=out[2]=matte;out[3]=1;}
}
__global__ void TemporalFeatures(const float* source, float* destination, const float* temporal) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 5184 * 256) return;
    const int channel = index % 256, pixel = index / 256;
    destination[index] = source[channel * 5184 + pixel] + (temporal ? temporal[channel] : 0.f);
}
__global__ void Horizontal(const uint8_t* source, int width, int height, ptrdiff_t stride, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= height * Size * 3) return;
    const int c = i % 3, x = (i / 3) % Size, y = i / (Size * 3);
    const float scale = float(width) / Size, support = fmaxf(scale, 1.f);
    const float center = (x + .5f) * scale;
    const int first = max(0, int(floorf(center - support + .5f)));
    const int last = min(width, int(floorf(center + support + .5f)));
    float sum = 0, weight = 0;
    for (int sx = first; sx < last; ++sx) {
        const float w = fmaxf(0.f, 1.f - fabsf((sx + .5f - center) / support));
        sum += source[y * stride + sx * 3 + c] * w; weight += w;
    }
    output[i] = sum / weight;
}
__global__ void Vertical(const float* source, int height, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Size * Size * 3) return;
    const int x = i % Size, y = (i / Size) % Size, c = i / (Size * Size);
    const float scale = float(height) / Size, support = fmaxf(scale, 1.f);
    const float center = (y + .5f) * scale;
    const int first = max(0, int(floorf(center - support + .5f)));
    const int last = min(height, int(floorf(center + support + .5f)));
    float sum = 0, weight = 0;
    for (int sy = first; sy < last; ++sy) {
        const float w = fmaxf(0.f, 1.f - fabsf((sy + .5f - center) / support));
        sum += source[(sy * Size + x) * 3 + c] * w; weight += w;
    }
    output[i] = float(__float2int_rn(fminf(255.f, fmaxf(0.f, sum / weight)))) / 127.5f - 1.f;
}
__global__ void ResizeMask(const float* source, int mw, int mh, float* output, int width, int height, bool binary) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    const float x = fmaxf(0.f, (i % width + .5f) * mw / width - .5f);
    const float y = fmaxf(0.f, (i / width + .5f) * mh / height - .5f);
    const int x0 = min(int(x), mw - 1), x1 = min(x0 + 1, mw - 1);
    const int y0 = min(int(y), mh - 1), y1 = min(y0 + 1, mh - 1);
    const float a = source[y0 * mw + x0] * (1.f - (x - x0)) + source[y0 * mw + x1] * (x - x0);
    const float b = source[y1 * mw + x0] * (1.f - (x - x0)) + source[y1 * mw + x1] * (x - x0);
    const float value = a * (1.f - (y - y0)) + b * (y - y0);
    output[i] = binary ? (value > 0 ? 1.f : 0.f) : value >= 0 ? 1.f / (1.f + expf(-value)) : expf(value) / (1.f + expf(value));
}
}
__global__ void Rasterize(MaskRasterInputs inputs, size_t pixels, uint32_t* labels, uint8_t* combined, uint8_t* individual) {
    const size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float best = .5f;
    uint32_t id = 0;
    for (int i = 0; i < inputs.count; ++i) {
        const float probability = inputs.probabilities[i][p];
        individual[size_t(i) * pixels + p] = probability > .5f ? 255 : 0;
        if (probability > best || (probability == best && probability > .5f && inputs.ids[i] < id)) {
            best = probability; id = inputs.ids[i];
        }
    }
    labels[p] = id; combined[p] = id ? 255 : 0;
}
__global__ void Bounds(const float* probabilities, int width, int height, float* output) {
    __shared__ int bounds[5];
    if (threadIdx.x == 0) {bounds[0]=width; bounds[1]=height; bounds[2]=0; bounds[3]=0; bounds[4]=0;}
    __syncthreads();
    int xmin=width, ymin=height, xmax=0, ymax=0, invalid=0;
    for (int p=threadIdx.x; p<width*height; p+=blockDim.x) {
        const float v=probabilities[p];
        if (!isfinite(v) || v<0 || v>1) invalid=1;
        if (v>.5f) {const int x=p%width, y=p/width; xmin=min(xmin,x); ymin=min(ymin,y); xmax=max(xmax,x+1); ymax=max(ymax,y+1);}
    }
    atomicMin(bounds,xmin); atomicMin(bounds+1,ymin); atomicMax(bounds+2,xmax); atomicMax(bounds+3,ymax); atomicMax(bounds+4,invalid);
    __syncthreads();
    if (threadIdx.x==0) {
        for(int i=0;i<4;++i) output[i]=bounds[2]>0?float(bounds[i]):0.f;
        output[4]=float(bounds[4]);
    }
}
void MaskBounds(const float* probabilities, int width, int height, float* bounds, void* stream) {
    TraceRange trace{"kernel.Bounds"};
    Bounds<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(probabilities,width,height,bounds);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void RasterizeMasks(MaskRasterInputs inputs, size_t pixels, uint32_t* labels,
                    uint8_t* combined, uint8_t* individual, void* stream) {
    TraceRange trace{"kernel.Rasterize"};
    Rasterize<<<(pixels + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(inputs, pixels, labels, combined, individual);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void PreprocessDeviceRgb(const uint8_t* input, int width, int height, std::ptrdiff_t row_bytes,
                         float* scratch, float* output, void* stream) {
    TraceRange horizontal{"kernel.Horizontal"};
    Horizontal<<<(height * Size * 3 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(input, width, height, row_bytes, scratch);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    horizontal.Stop();
    TraceRange vertical{"kernel.Vertical"};
    Vertical<<<(Size * Size * 3 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(scratch, height, output);
    status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void ResizeMaskProbability(const float* logits, int mask_width, int mask_height,
                           float* probabilities, int width, int height, void* stream, bool binary) {
    TraceRange trace{"kernel.ResizeMask"};
    ResizeMask<<<(width * height + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, mask_width, mask_height, probabilities, width, height, binary);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void PackTemporalFeatures(const float* source, float* destination, const float* temporal, void* stream) {
    TraceRange trace{"kernel.TemporalFeatures"};
    TemporalFeatures<<<(5184 * 256 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(source, destination, temporal);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void SelectQueries(const float* classes,const float* presence,int queries,float threshold,
                   int* selected,float* valid,int* error,void* stream) {
    TraceRange trace{"kernel.Select"};
    Select<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(classes,presence,queries,threshold,selected,valid,error);
    const auto s=cudaGetLastError();if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void GatherMasks(const float* masks,const int* selected,float* packed,void* stream) {
    TraceRange trace{"kernel.Gather"};
    Gather<<<(16*288*288+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(masks,selected,packed);
    const auto s=cudaGetLastError();if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void MaskPointers(float* pointers,const float* valid,void* stream) {
    TraceRange trace{"kernel.ClearPointers"};
    ClearPointers<<<16,256,0,static_cast<cudaStream_t>(stream)>>>(pointers,valid);
    const auto s=cudaGetLastError();if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void PointerPosition(float distance,const float* weights,const float* bias,float* output,void* stream) {
    TraceRange trace{"kernel.Position"};
    Position<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(distance,weights,bias,output);
    const auto s=cudaGetLastError();if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void WriteRgbaMasks(const vision::DeviceRgbaView& source,const vision::DeviceOutputView& output,
                    const float* masks,const float* valid,const float* logits,int selected,bool labels,void* stream) {
    TraceRange trace{"kernel.RgbaMasks"};
    if(output.width<=0 || output.height<=0) return;
    RgbaMasks<<<dim3((output.width+15)/16,(output.height+15)/16),dim3(16,16),0,static_cast<cudaStream_t>(stream)>>>(source,output,masks,valid,logits,selected,labels);
    const auto s=cudaGetLastError();if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
}
