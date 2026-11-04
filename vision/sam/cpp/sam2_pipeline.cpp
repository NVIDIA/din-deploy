// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam2_pipeline.h"
#include "sam2_io.h"
#include "sam.h"
#include "sam_device.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

namespace din::sam2 {
namespace {
void Check(cudaError_t status) {if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
sam::SamConfig RuntimeConfig(const Sam2PipelineConfig& config) {
    sam::SamConfig result;
    result.model=sam::ModelFamily::Sam2;result.model_dir=config.model_dir;
    result.cache_dir=config.ep_cache_dir;result.provider=config.provider;
    result.state_capacity=config.propagate ? 9:1;
    const auto metadata=LoadSam2Config(config.model_dir/"metadata.json");
    result.max_source_width=result.max_source_height=int(metadata.max_input_size);
    if(config.provider=="trt-rtx") Check(cudaGetDevice(&result.device));
    return result;
}
constexpr size_t MaskPixels=256*256,MaskSlots=128,CheckpointSlots=8;
float Sample(const float* data,int x,int y,int width,int height) {
    const float fx=std::max(0.f,(x+.5f)*256/width-.5f),fy=std::max(0.f,(y+.5f)*256/height-.5f);
    const int x0=std::min(int(fx),255),x1=std::min(x0+1,255),y0=std::min(int(fy),255),y1=std::min(y0+1,255);
    const float a=data[y0*256+x0]*(1-(fx-x0))+data[y0*256+x1]*(fx-x0);
    const float b=data[y1*256+x0]*(1-(fx-x0))+data[y1*256+x1]*(fx-x0);
    return a*(1-(fy-y0))+b*(fy-y0);
}
}
struct Sam2Pipeline::Impl {
    Sam2PipelineConfig config;
    sam::Sam runtime;
    sam::StateLease live;
    struct Checkpoint {sam::StateLease state;int64_t frame=0;bool valid=false;};
    std::array<Checkpoint,CheckpointSlots> checkpoints;
    std::array<std::optional<int64_t>,MaskSlots> cached{};
    sam::State* previous=nullptr;
    sam::PreparedPrompt prepared;
    Prompt prompt;
    sam::CompletionToken last;
    bool prompt_loaded=false,ready=false,gpu=false;
    size_t next_mask=0,next_checkpoint=0;
    int device=0;
    void* device_mask=nullptr;
    void* device_cache=nullptr;
    Ort::Float16_t* pinned=nullptr;
    std::vector<float> host_mask,host_cache;
    cudaEvent_t input=nullptr,output=nullptr;
    explicit Impl(Sam2PipelineConfig cfg):config(std::move(cfg)),runtime(RuntimeConfig(config)),
        live(runtime.AcquireState()),gpu(config.provider=="trt-rtx"),host_mask(MaskPixels) {
        if(config.propagate) for(auto& cp:checkpoints) cp.state=runtime.AcquireState();
        if(gpu) {
            Check(cudaGetDevice(&device));
            try {
                Check(cudaMalloc(&device_mask,MaskPixels*2));
                Check(cudaMalloc(&device_cache,MaskSlots*MaskPixels*2));
                Check(cudaMallocHost(reinterpret_cast<void**>(&pinned),MaskPixels*2));
                Check(cudaEventCreateWithFlags(&input,cudaEventDisableTiming));
                Check(cudaEventCreateWithFlags(&output,cudaEventDisableTiming));
            } catch(...) {Release();throw;}
        } else host_cache.resize(MaskSlots*MaskPixels);
    }
    ~Impl() {if(gpu) cudaStreamSynchronize(Stream());Release();}
    void Release() {if(!gpu) return;if(input) cudaEventDestroy(input);if(output) cudaEventDestroy(output);cudaFree(device_mask);cudaFree(device_cache);cudaFreeHost(pinned);}
    cudaStream_t Stream() const {return static_cast<cudaStream_t>(runtime.Stream());}
    void Prepare(const Prompt& p) {
        if(!prompt_loaded || prompt.coords!=p.coords || prompt.labels!=p.labels) {
            sam::Prompt value;value.coordinates=p.coords;value.labels=p.labels;
            prepared=runtime.PreparePrompt(value);prompt=p;prompt_loaded=true;
        }
    }
    void WaitInput(void* host) {Check(cudaEventRecord(input,static_cast<cudaStream_t>(host)));Check(cudaStreamWaitEvent(Stream(),input,0));}
    void Handoff(void* host) {
        Check(cudaEventRecord(output,Stream()));
        if(host) Check(cudaStreamWaitEvent(static_cast<cudaStream_t>(host),output,0));
        else {Check(cudaEventSynchronize(output));last.Wait();}
    }
    sam::MaskOutputView Target() {
        sam::MaskOutputView target;
        target.data=gpu ? device_mask:host_mask.data();target.width=target.height=256;
        target.memory=gpu ? sam::MemoryLocation::Device:sam::MemoryLocation::Host;
        target.type=gpu ? sam::ScalarType::Float16:sam::ScalarType::Float32;target.device=device;
        return target;
    }
    void Run(const sam::FrameView& frame,bool tracking) {
        sam::Initialization init{prepared,std::numeric_limits<size_t>::max(),tracking && config.propagate};
        auto submitted=runtime.RunFrame(frame,Target(),*live,tracking ? previous:nullptr,&init);
        last=std::move(submitted.completion);previous=tracking ? live.get():nullptr;
    }
    size_t Find(int64_t frame) const {for(size_t i=0;i<MaskSlots;++i) if(cached[i]==frame) return i;return MaskSlots;}
    Image Read(const Image& dimensions,const void* data) {
        if(!dimensions.width || !dimensions.height) throw std::invalid_argument("Invalid mask dimensions");
        if(gpu) {
            Check(cudaMemcpyAsync(pinned,data,MaskPixels*2,cudaMemcpyDeviceToHost,Stream()));
            Check(cudaStreamSynchronize(Stream()));
            for(size_t i=0;i<MaskPixels;++i) host_mask[i]=pinned[i].ToFloat();
        } else if(data!=host_mask.data()) std::memcpy(host_mask.data(),data,MaskPixels*4);
        Image mask{dimensions.width,dimensions.height,{}};
        mask.rgb.resize(size_t(mask.width)*mask.height*3);
        for(int y=0;y<int(mask.height);++y) for(int x=0;x<int(mask.width);++x) {
            const uint8_t value=Sample(host_mask.data(),x,y,int(mask.width),int(mask.height))>0 ? 255:0;
            const auto pixel=(size_t(y)*mask.width+x)*3;
            mask.rgb[pixel]=mask.rgb[pixel+1]=mask.rgb[pixel+2]=value;
        }
        return mask;
    }
};
Sam2Pipeline::Sam2Pipeline(Sam2PipelineConfig config):impl_(std::make_unique<Impl>(std::move(config))) {}
Sam2Pipeline::~Sam2Pipeline()=default;
Image Sam2Pipeline::SegmentImage(const Image& image,const Prompt& prompt) {
    impl_->Prepare(prompt);impl_->ready=false;
    sam::FrameView frame{image.rgb.data(),int(image.width),int(image.height),ptrdiff_t(image.width)*3,sam::MemoryLocation::Host};
    if(image.rgb.size()!=size_t(image.width)*image.height*3) throw std::invalid_argument("Invalid packed SAM2 image");
    impl_->Run(frame,false);impl_->last.Wait();
    return impl_->Read(image,impl_->gpu ? impl_->device_mask:impl_->host_mask.data());
}
void Sam2Pipeline::ResetSequence(const Prompt& prompt) {
    impl_->Prepare(prompt);impl_->previous=nullptr;impl_->ready=true;
}
Image Sam2Pipeline::ProcessNextFrame(const Image& image,bool produce_mask) {
    if(!impl_->ready) throw std::logic_error("ResetSequence must precede ProcessNextFrame");
    if(image.rgb.size()!=size_t(image.width)*image.height*3) throw std::invalid_argument("Invalid packed SAM2 image");
    sam::FrameView frame{image.rgb.data(),int(image.width),int(image.height),ptrdiff_t(image.width)*3,sam::MemoryLocation::Host};
    impl_->Run(frame,true);impl_->last.Wait();
    return produce_mask ? impl_->Read(image,impl_->gpu ? impl_->device_mask:impl_->host_mask.data()):Image{};
}
void Sam2Pipeline::ProcessDeviceFrame(const DeviceRgbaView& source,const DeviceOutputView* output,void* stream,const Prompt* image_prompt) {
    auto& p=*impl_;if(!p.gpu) throw std::logic_error("Device frames require a GPU provider");
    sam::detail::ValidateRgba(source,output,p.device);
    if(image_prompt) {p.Prepare(*image_prompt);p.ready=false;}
    else if(!p.ready) throw std::logic_error("ResetSequence must precede CUDA tracking");
    p.WaitInput(stream);
    sam::FrameView frame;
    frame.data=source.data;frame.width=source.width;frame.height=source.height;frame.row_bytes=source.row_bytes;
    frame.format=sam::FrameFormat::Rgba32F;frame.origin=sam::Origin::BottomLeft;
    frame.premultiplied=source.premultiplied;frame.device=p.device;
    p.Run(frame,image_prompt==nullptr);
    if(output) WriteDeviceMatte(source,*output,p.device_mask,256,p.Stream());
    p.Handoff(stream);
}
void Sam2Pipeline::ClearTemporalCache() {
    impl_->cached.fill(std::nullopt);for(auto& cp:impl_->checkpoints) cp.valid=false;
}
void Sam2Pipeline::RememberTemporalFrame(int64_t frame,bool checkpoint) {
    auto& p=*impl_;
    if(!p.ready || !p.config.propagate || !p.previous) throw std::logic_error("Temporal caching requires a running tracker");
    size_t slot=p.Find(frame);if(slot==MaskSlots) slot=p.next_mask++%MaskSlots;
    if(p.gpu) Check(cudaMemcpyAsync(static_cast<char*>(p.device_cache)+slot*MaskPixels*2,p.device_mask,MaskPixels*2,cudaMemcpyDeviceToDevice,p.Stream()));
    else std::copy_n(p.host_mask.data(),MaskPixels,p.host_cache.data()+slot*MaskPixels);
    p.cached[slot]=frame;
    if(checkpoint) {
        if(p.previous!=p.live.get()) {
            for(auto& cp:p.checkpoints) if(cp.state.get()==p.previous) {cp.frame=frame;cp.valid=true;return;}
        }
        auto& cp=p.checkpoints[p.next_checkpoint++%CheckpointSlots];
        // Retain the current state without a snapshot copy. The next frame uses
        // another reserved destination and branches from this preserved handle.
        std::swap(cp.state,p.live);cp.frame=frame;cp.valid=true;p.previous=cp.state.get();
    }
}
bool Sam2Pipeline::HasTemporalMask(int64_t frame) const {return impl_->Find(frame)!=MaskSlots;}
bool Sam2Pipeline::RestoreTemporalCheckpoint(int64_t before,int64_t& restored) {
    Impl::Checkpoint* best=nullptr;
    for(auto& cp:impl_->checkpoints) if(cp.valid && cp.frame<before && (!best || cp.frame>best->frame)) best=&cp;
    if(!best) return false;
    impl_->previous=best->state.get();impl_->ready=true;restored=best->frame;return true;
}
void Sam2Pipeline::RenderTemporalMask(int64_t frame,const DeviceRgbaView& source,const DeviceOutputView& output,void* stream) {
    auto& p=*impl_;const auto slot=p.Find(frame);
    if(!p.gpu || slot==MaskSlots) throw std::logic_error("Temporal mask was evicted or is not on GPU");
    sam::detail::ValidateRgba(source,&output,p.device);
    p.WaitInput(stream);
    WriteDeviceMatte(source,output,static_cast<const char*>(p.device_cache)+slot*MaskPixels*2,256,p.Stream());
    p.Handoff(stream);
}
Image Sam2Pipeline::ReadTemporalMask(int64_t frame,const Image& dimensions) {
    auto& p=*impl_;const auto slot=p.Find(frame);
    if(slot==MaskSlots) throw std::logic_error("Temporal mask was evicted");
    return p.Read(dimensions,p.gpu ? static_cast<const char*>(p.device_cache)+slot*MaskPixels*2
                                  :static_cast<const void*>(p.host_cache.data()+slot*MaskPixels));
}
void Sam2Pipeline::RunBatch(const std::vector<std::filesystem::path>& frames,const Prompt& prompt,const std::filesystem::path& output) {
    if(frames.empty()) throw std::invalid_argument("No input frames");
    ResetSequence(prompt);if(impl_->config.dump_masks) std::filesystem::create_directories(output);
    detail::AsyncMaskWriter writer(impl_->config.dump_masks);
    auto next=std::async(std::launch::async,[&]{return LoadPng(frames.front());});
    for(size_t i=0;i<frames.size();++i) {
        auto image=next.get();
        if(i+1<frames.size()) next=std::async(std::launch::async,[&,index=i+1]{return LoadPng(frames[index]);});
        auto mask=ProcessNextFrame(image,impl_->config.dump_masks);
        if(impl_->config.dump_masks) writer.EnqueueRgb(output/FrameMaskName(i,"_fast.png"),mask.rgb.data(),mask.width,mask.height);
    }
    writer.Finish();std::cout<<"Processed "<<frames.size()<<" frames\n";
}
} // namespace din::sam2
