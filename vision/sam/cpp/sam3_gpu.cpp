// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_gpu.h"
#include "sam3_cuda.h"
#include "sam.h"
#include "sam_device.h"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace din::sam3 {
namespace {
void Check(cudaError_t status) {if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
sam::SamConfig RuntimeConfig(const Sam3PipelineConfig& config) {
    if(config.provider!="trt-rtx") throw std::invalid_argument("SAM3 GPU adapter requires trt-rtx");
    sam::SamConfig result;
    result.model=sam::ModelFamily::Sam3;result.model_dir=config.model_dir;result.cache_dir=config.cache_dir;
    result.provider=config.provider;result.confidence_threshold=config.confidence_threshold;
    result.diagnostic_dir=config.diagnostic_dir;result.diagnostic_graph_dir=config.diagnostic_graph_dir;
    result.diagnostic_decoder=config.diagnostic_decoder;
    result.max_source_width=result.max_source_height=8192;
    Check(cudaGetDevice(&result.device));return result;
}
constexpr size_t CacheSlots=12,MaskValues=16*288*288;
}
struct Sam3GpuPipeline::Impl {
    sam::Sam runtime;
    sam::StateLease state;
    sam::PreparedPrompt prepared;
    sam::Initialization initialization;
    sam::CompletionToken last;
    std::string text;
    bool temporal=false,started=false;
    size_t count=0,next_cache=0;
    int width=0,height=0,device=0;
    float* cache=nullptr;
    float* pinned=nullptr;
    cudaEvent_t input=nullptr,output=nullptr;
    std::array<std::optional<int64_t>,CacheSlots> cached{};
    explicit Impl(const Sam3PipelineConfig& config):runtime(RuntimeConfig(config)),state(runtime.AcquireState()) {
        Check(cudaGetDevice(&device));
        try {
            Check(cudaMalloc(reinterpret_cast<void**>(&cache),CacheSlots*SavedMask::Values*sizeof(float)));
            Check(cudaMallocHost(reinterpret_cast<void**>(&pinned),SavedMask::Values*sizeof(float)));
            Check(cudaEventCreateWithFlags(&input,cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&output,cudaEventDisableTiming));
        } catch(...) {Release();throw;}
    }
    ~Impl() {cudaStreamSynchronize(Stream());Release();}
    void Release() {if(input) cudaEventDestroy(input);if(output) cudaEventDestroy(output);cudaFree(cache);cudaFreeHost(pinned);}
    cudaStream_t Stream() const {return static_cast<cudaStream_t>(runtime.Stream());}
    void WaitInput(void* host) {Check(cudaEventRecord(input,static_cast<cudaStream_t>(host)));Check(cudaStreamWaitEvent(Stream(),input,0));}
    void Handoff(void* host) {
        Check(cudaEventRecord(output,Stream()));
        if(host) Check(cudaStreamWaitEvent(static_cast<cudaStream_t>(host),output,0));
        else {Check(cudaEventSynchronize(output));last.Wait();}
    }
    size_t Find(int64_t frame) const {for(size_t i=0;i<CacheSlots;++i) if(cached[i]==frame) return i;return CacheSlots;}
    void Reset() {started=false;cached.fill(std::nullopt);next_cache=0;}
    void Configure(const TextPrompt& prompt,bool track,size_t frames) {
        if(text==prompt.text && temporal==track && count==frames) return;
        if(track && !frames) throw std::invalid_argument("SAM3 tracking needs a positive duration");
        last.Wait();
        prepared=runtime.PreparePrompt({.text=prompt.text});text=prompt.text;temporal=track;count=frames;
        initialization={prepared,std::max<size_t>(1,frames),track};Reset();
    }
    void Submit(const vision::DeviceRgbaView& source,void* host,int64_t frame) {
        if(text.empty()) throw std::logic_error("Configure SAM3 before submission");
        last.Ready(); // Reports completed deferred diagnostics without waiting.
        if(!temporal && (source.width!=width || source.height!=height)) cached.fill(std::nullopt);
        WaitInput(host);
        size_t slot=Find(frame);if(slot==CacheSlots) slot=next_cache++%CacheSlots;
        float* destination=cache+slot*SavedMask::Values;
        sam::FrameView input_frame;
        input_frame.data=source.data;input_frame.width=source.width;input_frame.height=source.height;
        input_frame.row_bytes=source.row_bytes;input_frame.format=sam::FrameFormat::Rgba32F;
        input_frame.origin=sam::Origin::BottomLeft;input_frame.premultiplied=source.premultiplied;input_frame.device=device;
        sam::MaskOutputView masks;
        masks.data=destination;masks.width=masks.height=288;masks.validity=destination+MaskValues;
        masks.presence_logits=destination+MaskValues+16;masks.device=device;
        auto submitted=runtime.RunFrame(input_frame,masks,*state,temporal && started ? state.get():nullptr,&initialization);
        last=std::move(submitted.completion);
        width=source.width;height=source.height;started=true;cached[slot]=frame;Handoff(host);
    }
};
Sam3GpuPipeline::Sam3GpuPipeline(Sam3PipelineConfig config):impl_(std::make_unique<Impl>(config)) {}
Sam3GpuPipeline::~Sam3GpuPipeline() {
    int previous=0;cudaGetDevice(&previous);cudaSetDevice(impl_->device);impl_.reset();cudaSetDevice(previous);
}
void Sam3GpuPipeline::Configure(const TextPrompt& prompt,bool temporal,size_t count) {impl_->Configure(prompt,temporal,count);}
void Sam3GpuPipeline::Reset() {impl_->Reset();}
void Sam3GpuPipeline::Submit(const vision::DeviceRgbaView& source,void* stream,int64_t frame) {impl_->Submit(source,stream,frame);}
bool Sam3GpuPipeline::HasFrame(int64_t frame) const {return impl_->Find(frame)!=CacheSlots;}
void Sam3GpuPipeline::Render(int64_t frame,const vision::DeviceRgbaView& source,const vision::DeviceOutputView& output,
                            void* stream,int selected,bool labels) {
    auto& p=*impl_;const auto slot=p.Find(frame);
    sam::detail::ValidateRgba(source,&output,p.device);
    if(slot==CacheSlots || selected<0 || selected>16 || source.width!=p.width || source.height!=p.height ||
       !source.data || !output.data || output.row_bytes==PTRDIFF_MIN ||
       std::abs(output.row_bytes)<ptrdiff_t(output.width)*16) throw std::invalid_argument("Invalid cached SAM3 render");
    p.WaitInput(stream);
    const float* masks=p.cache+slot*SavedMask::Values;
    WriteRgbaMasks(source,output,masks,masks+MaskValues,p.temporal ? masks+MaskValues+16:nullptr,selected,labels,p.Stream());
    p.Handoff(stream);
}
void Sam3GpuPipeline::Synchronize() {Check(cudaStreamSynchronize(impl_->Stream()));impl_->last.Wait();}
SavedMask Sam3GpuPipeline::Export(int64_t frame) {
    auto& p=*impl_;const auto slot=p.Find(frame);
    if(slot==CacheSlots) throw std::invalid_argument("SAM3 frame is not cached");
    Check(cudaMemcpyAsync(p.pinned,p.cache+slot*SavedMask::Values,SavedMask::Values*sizeof(float),cudaMemcpyDeviceToHost,p.Stream()));
    Synchronize();
    SavedMask result{p.width,p.height,p.temporal,{}};
    result.values.assign(p.pinned,p.pinned+SavedMask::Values);return result;
}

struct SavedMaskRenderer::Impl {
    cudaStream_t stream=nullptr;
    cudaEvent_t input=nullptr,done=nullptr,uploaded=nullptr;
    float* pinned=nullptr;
    void* device=nullptr;
    int ordinal=0;
    bool pending=false;
    Impl() {
        try {
            Check(cudaGetDevice(&ordinal));
            Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            Check(cudaEventCreateWithFlags(&input,cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&uploaded,cudaEventDisableTiming));
            Check(cudaMallocHost(reinterpret_cast<void**>(&pinned),SavedMask::Values*sizeof(float)));
            Check(cudaMalloc(&device,SavedMask::Values*sizeof(float)));
        } catch(...) {Release();throw;}
    }
    void Release() {
        if(stream) cudaStreamSynchronize(stream);
        cudaFree(device);cudaFreeHost(pinned);
        if(input) cudaEventDestroy(input);if(done) cudaEventDestroy(done);
        if(uploaded) cudaEventDestroy(uploaded);
        if(stream) cudaStreamDestroy(stream);
    }
    ~Impl() {Release();}
};
SavedMaskRenderer::SavedMaskRenderer():impl_(std::make_unique<Impl>()) {}
SavedMaskRenderer::~SavedMaskRenderer() {
    int previous=0;cudaGetDevice(&previous);cudaSetDevice(impl_->ordinal);
    impl_.reset();cudaSetDevice(previous);
}
void SavedMaskRenderer::Render(const SavedMask& mask,const vision::DeviceRgbaView& src,
                              const vision::DeviceOutputView& out,void* host,int selected,bool labels) {
    auto& p=*impl_;
    int device;Check(cudaGetDevice(&device));
    if(device!=p.ordinal || mask.values.size()!=SavedMask::Values || mask.width!=src.width || mask.height!=src.height ||
       selected<0 || selected>16 || !src.data || !out.data || src.width<=0 || src.height<=0 ||
       src.row_bytes==PTRDIFF_MIN || std::abs(src.row_bytes)<ptrdiff_t(src.width)*16 ||
       out.row_bytes==PTRDIFF_MIN || out.width<0 || out.height<0 || std::abs(out.row_bytes)<ptrdiff_t(out.width)*16)
        throw std::invalid_argument("Invalid SAM3 saved mask or output geometry");
    // Wait only before reusing pinned staging from the preceding upload.
    if(p.pending) Check(cudaEventSynchronize(p.uploaded));
    std::copy(mask.values.begin(),mask.values.end(),p.pinned);
    auto stream=static_cast<cudaStream_t>(host);
    try {
        Check(cudaEventRecord(p.input,stream));Check(cudaStreamWaitEvent(p.stream,p.input,0));
        Check(cudaMemcpyAsync(p.device,p.pinned,SavedMask::Values*sizeof(float),cudaMemcpyHostToDevice,p.stream));
        Check(cudaEventRecord(p.uploaded,p.stream));
        auto* data=static_cast<float*>(p.device);
        WriteRgbaMasks(src,out,data,data+16*288*288,mask.temporal?data+16*288*288+16:nullptr,selected,labels,p.stream);
        Check(cudaEventRecord(p.done,p.stream));p.pending=true;
        if(host) Check(cudaStreamWaitEvent(stream,p.done,0));
        else Check(cudaEventSynchronize(p.done));
    } catch(...) {cudaStreamSynchronize(p.stream);throw;}
}
}
