// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include "sam.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace din::sam3 {
namespace {
sam::SamConfig RuntimeConfig(const Sam3PipelineConfig& config) {
    sam::SamConfig result;
    result.model=sam::ModelFamily::Sam3;
    result.model_dir=config.model_dir;result.cache_dir=config.cache_dir;
    result.provider=config.provider;result.confidence_threshold=config.confidence_threshold;
    result.diagnostic_dir=config.diagnostic_dir;result.diagnostic_graph_dir=config.diagnostic_graph_dir;
    result.diagnostic_decoder=config.diagnostic_decoder;
    result.max_source_width=result.max_source_height=8192;
    if(result.provider=="trt-rtx") cudaGetDevice(&result.device);
    return result;
}
}
struct Sam3Pipeline::Impl {
    Sam3PipelineConfig config;
    sam::Sam runtime;
    sam::StateLease state;
    sam::PreparedPrompt prompt;
    std::string text;
    size_t count=1;
    bool sequence=false,started=false;
    std::map<std::string,double> timings;
    RasterizedMasks rasters;
    cudaEvent_t producer=nullptr;
    explicit Impl(Sam3PipelineConfig cfg):config(std::move(cfg)),runtime(RuntimeConfig(config)),state(runtime.AcquireState()) {
        if(config.provider=="trt-rtx") {
            const auto status=cudaEventCreateWithFlags(&producer,cudaEventDisableTiming);
            if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
    }
    ~Impl() {if(producer) cudaEventDestroy(producer);}
    void Prepare(const TextPrompt& p) {
        if(text!=p.text) {prompt=runtime.PreparePrompt({.text=p.text});text=p.text;}
    }
    FrameResult Run(const Image& image,const DeviceFrame* device) {
        if(!image.width || !image.height || image.width>8192 || image.height>8192 ||
           (device && device->row_bytes>size_t(PTRDIFF_MAX)/image.height))
            throw std::invalid_argument("Invalid SAM3 image dimensions or stride");
        if(!device && image.rgb.size()!=size_t(image.width)*image.height*3)
            throw std::invalid_argument("Invalid packed SAM3 RGB image");
        const auto start=std::chrono::steady_clock::now();
        sam::FrameView frame;
        frame.data=device ? device->rgb:image.rgb.data();
        frame.width=int(image.width);frame.height=int(image.height);
        frame.row_bytes=device ? ptrdiff_t(device->row_bytes):ptrdiff_t(image.width)*3;
        frame.memory=device ? sam::MemoryLocation::Device:sam::MemoryLocation::Host;
        if(config.provider=="trt-rtx") {
            cudaGetDevice(&frame.device);
            if(device) {
                auto status=cudaEventRecord(producer,static_cast<cudaStream_t>(device->producer_stream));
                if(status==cudaSuccess) status=cudaStreamWaitEvent(static_cast<cudaStream_t>(runtime.Stream()),producer,0);
                if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        const size_t pixels=size_t(image.width)*image.height;
        std::vector<float> probabilities(16*pixels);
        std::array<float,16> validity{},scores{};
        std::array<std::array<float,4>,16> detection_boxes{};
        sam::MaskOutputView output;
        output.data=probabilities.data();output.width=int(image.width);output.height=int(image.height);
        output.memory=sam::MemoryLocation::Host;output.format=sam::MaskFormat::Probabilities;
        output.validity=validity.data();output.scores=scores.data();
        output.detection_boxes=detection_boxes[0].data();
        sam::Initialization init{prompt,count,sequence};
        auto submitted=runtime.RunFrame(frame,output,*state,sequence && started ? state.get():nullptr,&init);
        submitted.completion.Wait();started=true;
        FrameResult result{image.width,image.height,{}};
        for(size_t i=0;i<16;++i) if(validity[i]>0) {
            Instance instance;instance.id=uint32_t(i+1);instance.score=scores[i];
            instance.probabilities.assign(probabilities.begin()+ptrdiff_t(i*pixels),probabilities.begin()+ptrdiff_t((i+1)*pixels));
            std::array<float,4> box{float(image.width),float(image.height),0,0};
            for(size_t p=0;p<pixels;++p) {
                const float value=instance.probabilities[p];
                if(!std::isfinite(value) || value<0 || value>1) throw std::runtime_error("SAM3 produced invalid probabilities");
                if(value>.5f) {
                    const float x=float(p%image.width),y=float(p/image.width);
                    box[0]=std::min(box[0],x);box[1]=std::min(box[1],y);
                    box[2]=std::max(box[2],x+1);box[3]=std::max(box[3],y+1);
                }
            }
            instance.box=sequence ? (box[2]>0 ? box:std::array<float,4>{}):detection_boxes[i];
            result.instances.push_back(std::move(instance));
        }
        if(config.rasterize_output) {
            rasters={};rasters.width=result.width;rasters.height=result.height;
            rasters.labels=result.Labels();rasters.combined=result.CombinedMask();
            for(const auto& instance:result.instances) {
                rasters.ids.push_back(instance.id);
                for(float p:instance.probabilities) rasters.individual.push_back(p>.5f ? 255:0);
            }
        }
        timings.clear();timings["frame_total_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return result;
    }
};
Sam3Pipeline::Sam3Pipeline(Sam3PipelineConfig config):impl_(std::make_unique<Impl>(std::move(config))) {}
Sam3Pipeline::~Sam3Pipeline()=default;
FrameResult Sam3Pipeline::SegmentImage(const Image& image,const TextPrompt& prompt) {
    impl_->Prepare(prompt);impl_->sequence=false;impl_->started=false;impl_->count=1;return impl_->Run(image,nullptr);
}
FrameResult Sam3Pipeline::SegmentDeviceFrame(const DeviceFrame& frame,const TextPrompt& prompt) {
    impl_->Prepare(prompt);impl_->sequence=false;impl_->started=false;impl_->count=1;
    return impl_->Run({frame.width,frame.height,{}},&frame);
}
void Sam3Pipeline::ResetSequence(const TextPrompt& prompt,size_t count) {
    if(!count) throw std::invalid_argument("Sequence requires a positive frame count");
    impl_->Prepare(prompt);impl_->sequence=true;impl_->started=false;impl_->count=count;
}
FrameResult Sam3Pipeline::ProcessNextFrame(const Image& image) {
    if(!impl_->sequence) throw std::logic_error("ResetSequence is required");
    return impl_->Run(image,nullptr);
}
FrameResult Sam3Pipeline::ProcessDeviceFrame(const DeviceFrame& frame) {
    if(!impl_->sequence) throw std::logic_error("ResetSequence is required");
    return impl_->Run({frame.width,frame.height,{}},&frame);
}
const std::map<std::string,double>& Sam3Pipeline::LastTimings() const {return impl_->timings;}
const RasterizedMasks* Sam3Pipeline::LastRasterizedMasks() const {return impl_->config.rasterize_output ? &impl_->rasters:nullptr;}
std::vector<uint32_t> FrameResult::Labels() const {

    if (!width || !height || width > 8192 || height > 8192 || instances.size() > 16)
        throw std::runtime_error("Invalid SAM3 result dimensions or capacity");
    std::vector<uint32_t> labels(size_t(width) * height, 0);
    std::vector<float> best(labels.size(), .5f);
    std::set<uint32_t> ids;
    for (const auto& instance : instances) {
        if (instance.id == 0 || !ids.insert(instance.id).second || instance.probabilities.size() != labels.size()) throw std::runtime_error("Invalid SAM3 instance result");
        for (size_t i = 0; i < labels.size(); ++i) {
            float probability = instance.probabilities[i];
            if (!std::isfinite(probability) || probability < 0 || probability > 1) throw std::runtime_error("Invalid mask probability");
            if (probability > best[i] || (probability == best[i] && probability > .5f && instance.id < labels[i])) {
                best[i] = probability; labels[i] = instance.id;
            }
        }
    }
    return labels;
}
std::vector<uint8_t> FrameResult::CombinedMask() const {

    const auto labels = Labels();
    std::vector<uint8_t> mask(labels.size());
    std::transform(labels.begin(), labels.end(), mask.begin(), [](uint32_t label) {return label ? 255 : 0;});
    return mask;
}
}
