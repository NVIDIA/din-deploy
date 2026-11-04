// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace din::sam::detail {
namespace {
void Check() { const auto e = cudaGetLastError(); if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__device__ float Sigmoid(float x) { return x >= 0 ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x)); }
__global__ void Convert(FrameView src, uint8_t* rgb) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= src.width || y >= src.height) return;
    const int sy = src.origin == Origin::TopLeft ? y : src.height - 1 - y;
    const auto* row = static_cast<const char*>(src.data) + sy * src.row_bytes;
    const size_t out = (size_t(y) * src.width + x) * 3;
    for(int c = 0; c < 3; ++c) {
        if(src.format == FrameFormat::Rgb8) rgb[out+c] = reinterpret_cast<const uint8_t*>(row)[3*x+c];
        else {
            const float* pixel = reinterpret_cast<const float*>(row) + 4*x;
            float v = pixel[c];
            if(src.premultiplied) v = pixel[3] > 0 ? v / pixel[3] : 0;
            rgb[out+c] = isfinite(v) ? uint8_t(roundf(fminf(1.f, fmaxf(0.f, v)) * 255.f)) : 0;
        }
    }
}
template<class T> __global__ void Preprocess(const uint8_t* src, int w, int h, T* dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= 1024 || y >= 1024) return;
    float fx = fmaxf(0.f, (x+.5f)*w/1024-.5f), fy = fmaxf(0.f, (y+.5f)*h/1024-.5f);
    int x0 = min(int(fx),w-1), x1 = min(x0+1,w-1), y0 = min(int(fy),h-1), y1 = min(y0+1,h-1);
    const float means[3]{.485f,.456f,.406f}, stds[3]{.229f,.224f,.225f};
    for(int c = 0;c < 3;++c) {
        float a = (src[(size_t(y0)*w+x0)*3+c]/255.f)*(1-(fx-x0)) + (src[(size_t(y0)*w+x1)*3+c]/255.f)*(fx-x0);
        float b = (src[(size_t(y1)*w+x0)*3+c]/255.f)*(1-(fx-x0)) + (src[(size_t(y1)*w+x1)*3+c]/255.f)*(fx-x0);
        dst[(size_t(c)*1024+y)*1024+x] = T((a*(1-(fy-y0))+b*(fy-y0)-means[c])/stds[c]);
    }
}
__device__ int HistorySlot(const TemporalMeta& m, int history_index) {
    if(history_index == 0) return 0;
    return ((m.next_slot-1-(m.history_count-1)+(history_index-1)+15)%15)+1;
}
__global__ void PackSpatial(const TemporalMeta* state, const float* ring, const float* embeddings,
                             float* mi, float* mip, float* mem, float* mp, bool* valid) {
    const auto& m = *state;
    const size_t n = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if(n >= 7*Feature) return;
    const int i = int(n/Feature), channel = int(n%256), pixel = int((n%Feature)/256);
    const int count = min(m.history_count,7);
    const bool active = i<count;
    if(n%Feature == 0) valid[i] = active;
    float a=0,b=0,c=0,d=0;
    if(active) {
        const int history_index = i==0 ? 0 : (m.history_count>7 ? m.history_count-7+i : i);
        const int slot = HistorySlot(m,history_index);
        const uint64_t age = m.step-m.frames[slot];
        const int t = age>=7 ? 6 : (i==0 ? 6-int(age) : int(age)-1);
        const size_t offset = size_t(slot)*Sam3Entry + size_t(channel)*5184 + pixel;
        a=ring[offset]; b=ring[offset+Feature]+embeddings[t*256+channel];
        c=ring[offset+2*Feature]; d=ring[offset+3*Feature]+embeddings[t*256+channel];
    }
    mi[n]=a; mip[n]=b; mem[n]=c; mp[n]=d;
}
__global__ void PackPointers(const TemporalMeta* state, const float* ring, const float* weights, const float* bias,
                            float* pointers, float* positions, bool* valid, float* object_valid) {
    const auto& m = *state;
    const int i=blockIdx.x, c=threadIdx.x;
    const bool active = i<m.history_count;
    if(c==0) valid[i]=active;
    if(i==0 && c<16) object_valid[c]=m.valid[c];
    float value=0;
    int slot=0;
    if(active) {
        slot=HistorySlot(m,i);
        float distance=float(m.step-m.frames[slot])/float(max(uint64_t(1),min(uint64_t(16),m.length)-1));
        value=bias ? bias[c] : 0.f;
        for(int j=0;j<(weights ? 256:1);++j) {
            const int k=weights ? j:c;
            const float phase=distance/powf(10000.f,2.f*((k%128)/2)/128.f);
            const float sine=k<128 ? sinf(phase):cosf(phase);
            value+=weights ? weights[c*256+j]*sine:sine;
        }
    }
    for(int o=0;o<16;++o) {
        const size_t n=(size_t(i)*16+o)*256+c;
        pointers[n]=active ? ring[size_t(slot)*Sam3Entry+4*Feature+o*256+c]:0;
        positions[n]=value;
    }
}
__global__ void Decide(const TemporalMeta* state, const float* seed_valid, Decision* result) {
    bool any=false;
    for(int i=0;i<16;++i) any = any || seed_valid[i]>0;
    result->seed=!state->initialized;
    result->commit=state->initialized || any;
    result->slot=state->initialized ? state->next_slot:0;
}
__global__ void CommitFeatures(const TemporalMeta* state, float* ring, const Decision* decision,
    const float* image, const float* position, const float* sm, const float* sp, const float* so,
    const float* sv, const float* tm, const float* tp, const float* to) {
    const size_t n=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=Sam3Entry || !decision->commit) return;
    const bool seed=decision->seed;
    float value;
    if(n<Feature) value=image[n];
    else if(n<2*Feature) value=position[n-Feature];
    else if(n<3*Feature) value=(seed?sm:tm)[n-2*Feature];
    else if(n<4*Feature) value=(seed?sp:tp)[n-3*Feature];
    else {
        const size_t p=n-4*Feature;
        const float active=seed ? sv[p/256]:state->valid[p/256];
        value=active>0 ? (seed?so:to)[p]:0;
    }
    ring[size_t(decision->slot)*Sam3Entry+n]=value;
}
__global__ void CommitMasks(const TemporalMeta* state, const Decision* decision,
    const float* sv, const float* sm, const float* sl, const float* tm, const float* tl,
    float* masks, float* valid, float* scores, float* logits) {
    const size_t n=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=16*288*288) return;
    int object=int(n/(288*288));
    float logit=decision->seed ? sl[object]:tl[object];
    float active=decision->seed ? sv[object]:state->valid[object];
    masks[n]=decision->commit && active>0 ? (decision->seed?sm:tm)[n]:-1024.f;
    if(n%(288*288)==0) {
        valid[object]=decision->commit && active>0 && logit>0 ? 1.f:0.f;
        scores[object]=decision->commit && active>0 ? Sigmoid(logit):0;
        logits[object]=decision->commit && active>0 ? logit:-1024.f;
    }
}
__global__ void Finish(TemporalMeta* state, const Decision* decision, const float* seed_valid) {
    if(decision->commit) {
        state->frames[decision->slot]=state->step;
        if(decision->seed) {
            for(int i=0;i<16;++i) state->valid[i]=seed_valid[i];
            state->initialized=1; state->history_count=1; state->next_slot=1;
        } else {
            state->history_count=min(state->history_count+1,16);
            state->next_slot=1+state->next_slot%15;
        }
    }
    ++state->step;
}
__global__ void Scores(const float* classes,const float* presence,const int* selected,const float* valid,float* scores,float* logits,
                       const float* boxes,float* output_boxes,int width,int height) {
    int i=threadIdx.x;
    if(i<16) {
        const float score=valid[i]>0 && selected[i]>=0 ? Sigmoid(classes[selected[i]])*Sigmoid(presence[0]):0;
        scores[i]=score;logits[i]=score<=0 ? -1024.f:score>=1 ? 1024.f:logf(score/(1-score));
        for(int k=0;k<4;++k) output_boxes[i*4+k]=valid[i]>0 && selected[i]>=0 ? boxes[selected[i]*4+k]*(k%2 ? height:width):0;
    }
}
__global__ void Result2(const void* logits,const void* presence,bool half,float* masks,float* valid,float* scores,float* output_logits) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<256*256) masks[i]=half ? __half2float(static_cast<const __half*>(logits)[i]):static_cast<const float*>(logits)[i];
    if(i==0) {
        const float score=half ? __half2float(*static_cast<const __half*>(presence)):*static_cast<const float*>(presence);
        valid[0]=1; scores[0]=Sigmoid(score);output_logits[0]=score;
    }
}
__global__ void Validity(bool* valid,int active) {int i=threadIdx.x;if(i<7) valid[i]=i<active;}
__global__ void Reset(TemporalMeta* meta,uint64_t length) {*meta=TemporalMeta{};meta->length=length;}
__device__ float Sample(const float* masks,int object,int size,int x,int y,int w,int h) {
    float fx=fmaxf(0.f,(x+.5f)*size/w-.5f),fy=fmaxf(0.f,(y+.5f)*size/h-.5f);
    int x0=min(int(fx),size-1),x1=min(x0+1,size-1),y0=min(int(fy),size-1),y1=min(y0+1,size-1);
    masks+=size_t(object)*size*size;
    float a=masks[y0*size+x0]*(1-(fx-x0))+masks[y0*size+x1]*(fx-x0);
    float b=masks[y1*size+x0]*(1-(fx-x0))+masks[y1*size+x1]*(fx-x0);
    return a*(1-(fy-y0))+b*(fy-y0);
}
__global__ void Output(const float* masks,const float* valid,const float* scores,const float* logits,const float* boxes,int objects,int size,MaskOutputView dst) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.z;
    if(x>=dst.width || y>=dst.height) return;
    int object=dst.selected_id ? dst.selected_id-1:o;
    const int element=dst.type==ScalarType::Float16 ? 2:dst.type==ScalarType::UInt8 ? 1:4;
    const ptrdiff_t row=dst.row_bytes ? dst.row_bytes:ptrdiff_t(dst.width)*element;
    const ptrdiff_t plane=dst.instance_bytes ? dst.instance_bytes:row*dst.height;
    char* pixel=static_cast<char*>(dst.data)+o*plane+y*row+x*element;
    float value=0;
    if(dst.format==MaskFormat::Labels) {
        float best=0;
        uint32_t id=0;
        for(int i=0;i<objects;++i) if(valid[i]>0 && (!dst.selected_id || i==object)) {
            float p=Sigmoid(Sample(masks,i,size,x,y,dst.width,dst.height));
            if(p>.5f && p>best) {best=p;id=i+1;}
        }
        *reinterpret_cast<uint32_t*>(pixel)=id;
    } else {
        float logit=Sample(masks,object,size,x,y,dst.width,dst.height);
        value=dst.format==MaskFormat::Logits ? logit:dst.format==MaskFormat::Probabilities ? Sigmoid(logit):float(logit>0);
        if(valid[object]<=0 && dst.format!=MaskFormat::Logits) value=0;
        if(dst.type==ScalarType::Float16) *reinterpret_cast<__half*>(pixel)=__float2half(value);
        else if(dst.type==ScalarType::UInt8) *reinterpret_cast<uint8_t*>(pixel)=uint8_t(value);
        else *reinterpret_cast<float*>(pixel)=value;
    }
    if(x==0 && y==0 && o==0) for(int i=0;i<objects;++i) {
        if(dst.validity) dst.validity[i]=valid[i];
        if(dst.scores) dst.scores[i]=scores[i];
        if(dst.ids) dst.ids[i]=valid[i]>0 ? i+1:0;
        if(dst.presence_logits) dst.presence_logits[i]=logits[i];
        if(dst.detection_boxes) for(int k=0;k<4;++k) dst.detection_boxes[i*4+k]=boxes[i*4+k];
    }
}
} // namespace
void ConvertFrame(const FrameView& v,uint8_t* rgb,void* stream) {
    Convert<<<dim3((v.width+15)/16,(v.height+15)/16),dim3(16,16),0,static_cast<cudaStream_t>(stream)>>>(v,rgb);Check();
}
void PreprocessSam2(const uint8_t* rgb,int w,int h,void* image,bool half,void* stream) {
    const dim3 grid(64,64),block(16,16);
    if(half) Preprocess<<<grid,block,0,static_cast<cudaStream_t>(stream)>>>(rgb,w,h,static_cast<__half*>(image));
    else Preprocess<<<grid,block,0,static_cast<cudaStream_t>(stream)>>>(rgb,w,h,static_cast<float*>(image));
    Check();
}
void PackSam3(const TemporalMeta* m,const float* ring,const float* embeddings,const float* weights,const float* bias,
    float* mi,float* mip,float* mem,float* mp,float* pointers,float* positions,bool* mv,bool* pv,float* ov,void* stream) {
    auto s=static_cast<cudaStream_t>(stream);
    PackSpatial<<<(7*Feature+255)/256,256,0,s>>>(m,ring,embeddings,mi,mip,mem,mp,mv);
    PackPointers<<<16,256,0,s>>>(m,ring,weights,bias,pointers,positions,pv,ov);Check();
}
void CommitSam3(TemporalMeta* m,float* ring,Decision* decision,const float* image,const float* pos,
    const float* sm,const float* sp,const float* so,const float* sv,const float* masks,const float* sl,
    const float* tm,const float* tp,const float* to,const float* tmask,const float* tl,
    float* output,float* valid,float* scores,float* logits,void* stream) {
    auto s=static_cast<cudaStream_t>(stream);
    Decide<<<1,1,0,s>>>(m,sv,decision);
    CommitFeatures<<<(Sam3Entry+255)/256,256,0,s>>>(m,ring,decision,image,pos,sm,sp,so,sv,tm,tp,to);
    CommitMasks<<<(16*288*288+255)/256,256,0,s>>>(m,decision,sv,masks,sl,tmask,tl,output,valid,scores,logits);
    Finish<<<1,1,0,s>>>(m,decision,sv);Check();
}
void DetectionScores(const float* classes,const float* presence,const int* selected,const float* valid,float* scores,float* logits,
                     const float* boxes,float* output_boxes,int width,int height,void* stream) {
    Scores<<<1,16,0,static_cast<cudaStream_t>(stream)>>>(classes,presence,selected,valid,scores,logits,boxes,output_boxes,width,height);Check();
}
void Sam2Result(const void* logits,const void* presence,bool half,float* masks,float* valid,float* scores,float* output_logits,void* stream) {
    Result2<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(logits,presence,half,masks,valid,scores,output_logits);Check();
}
void MemoryValidity(bool* valid,int active,void* stream) {
    Validity<<<1,7,0,static_cast<cudaStream_t>(stream)>>>(valid,active);Check();
}
void ResetMeta(TemporalMeta* meta,uint64_t length,void* stream) {Reset<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(meta,length);Check();}
void WriteMasks(const float* masks,const float* valid,const float* scores,const float* logits,const float* boxes,int objects,int size,const MaskOutputView& output,void* stream) {
    Output<<<dim3((output.width+15)/16,(output.height+15)/16,output.format==MaskFormat::Labels || output.selected_id ? 1:objects),dim3(16,16),0,static_cast<cudaStream_t>(stream)>>>(masks,valid,scores,logits,boxes,objects,size,output);Check();
}
} // namespace din::sam::detail
