// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam.h"
#include "sam_kernels.h"
#include "ort_session.h"
#include "sam3_cuda.h"
#include "sam3_tokenizer.h"
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace din::sam {
namespace {
using json = nlohmann::json;
using detail::TemporalMeta;
using detail::Feature;
using detail::Sam3Entry;
void Check(cudaError_t result) { if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
struct DeviceScope {
    int previous=0; bool restore=false;
    DeviceScope(bool gpu,int device) { if(gpu) {Check(cudaGetDevice(&previous)); if(previous!=device) {Check(cudaSetDevice(device));restore=true;}} }
    ~DeviceScope() {if(restore) cudaSetDevice(previous);}
};
struct OwnedStream {
    cudaStream_t value=nullptr;int device=0;
    ~OwnedStream() {if(value) {DeviceScope scope(true,device);cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
size_t TypeBytes(ONNXTensorElementDataType type) {
    switch(type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        default: throw std::runtime_error("Unsupported SAM tensor type");
    }
}
struct Buffer {
    void* data=nullptr;
    size_t bytes=0;
    bool gpu=false,pinned=false;
    int device=0;
    Buffer(size_t size,bool device_memory,int ordinal,bool pin=false):bytes(size),gpu(device_memory),pinned(pin),device(ordinal) {
        DeviceScope scope(gpu || pinned,device);
        if(gpu) Check(cudaMalloc(&data,bytes));
        else if(pinned) Check(cudaMallocHost(&data,bytes));
        else data=::operator new(bytes);
    }
    ~Buffer() {
        DeviceScope scope(gpu || pinned,device);
        if(gpu) cudaFree(data); else if(pinned) cudaFreeHost(data); else ::operator delete(data);
    }
    Buffer(const Buffer&)=delete;
    template<class T=float> T* As() const {return static_cast<T*>(data);}
};
struct Tensor {
    std::unique_ptr<Buffer> storage;
    Ort::Value value{nullptr};
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
    Tensor(std::vector<int64_t> dimensions,ONNXTensorElementDataType element,bool gpu,int device,size_t reserved_bytes=0)
        :shape(std::move(dimensions)),type(element) {
        size_t count=1;for(auto d:shape) {if(d<=0) throw std::runtime_error("SAM tensors must have static positive shapes");count*=size_t(d);}
        storage=std::make_unique<Buffer>(std::max(count*TypeBytes(type),reserved_bytes),gpu,device);
        const auto memory=gpu ? Ort::MemoryInfo("Cuda",OrtDeviceAllocator,device,OrtMemTypeDefault)
                              : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        value=Ort::Value::CreateTensor(memory,storage->data,storage->bytes,shape.data(),shape.size(),type);
    }
    void* Data() const {return storage->data;}
    template<class T=float> T* As() const {return storage->As<T>();}
};
struct Graph {
    std::unique_ptr<common::OrtRunner> runner;
    std::unique_ptr<Ort::IoBinding> binding;
    std::map<std::string,std::unique_ptr<Tensor>,std::less<>> outputs;
    Ort::RunOptions options;
    std::string name;
    cudaStream_t stream=nullptr;
    cudaGraphExec_t replay=nullptr;
    bool gpu=false;int device=0;
    Graph(const SamConfig& cfg,const std::string& graph,void* compute_stream):name(graph),stream(static_cast<cudaStream_t>(compute_stream)),gpu(cfg.provider=="trt-rtx"),device(cfg.device) {
        common::ModelProfile profile;
        profile.cache_subpath=graph;
        profile.embed_ep_context=false;
        // This layer captures the warmed enqueue itself. Do not nest the EP's
        // internal capture, whose signal nodes reject an already-capturing stream.
        profile.enable_cuda_graph=false;
        // The EP defaults to true when the option is absent. An explicit zero is
        // required before capturing its enqueue from our own persistent stream.
        profile.extra_ep_options.emplace_back("enable_cuda_graph","0");
        profile.has_user_compute_stream=cfg.provider=="trt-rtx";
        profile.user_compute_stream=compute_stream;
        profile.extra_ep_options.emplace_back("nv_persistent_context_memory","1");
        if(graph=="seed_memory") {
            // Preserve the validated dynamic antialias boundary in the engine
            // profile; ordinary initialization always binds the stable 1008 view.
            profile.min_shapes=profile.opt_shapes="mask_logits:16x1x1008x1008";
            profile.max_shapes="mask_logits:16x1x1152x1152";
        }
        auto path=cfg.model_dir/(graph+".onnx");
        if(!cfg.diagnostic_graph_dir.empty() && std::filesystem::exists(cfg.diagnostic_graph_dir/(graph+".onnx")))
            path=cfg.diagnostic_graph_dir/(graph+".onnx");
        if(graph=="tracking_decoder" && !cfg.diagnostic_decoder.empty()) path=cfg.diagnostic_decoder;
        const auto stamp=std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
        auto identity=path.parent_path().filename().string()+"_"+stamp;
        if(identity.size()>32) {
            // RTX writes additional engine/context suffixes. Keep Windows paths
            // below MAX_PATH even when a caller uses a nested cache directory.
            uint64_t hash=14695981039346656037ull;
            for(unsigned char c:std::filesystem::absolute(path).string()+stamp) {hash^=c;hash*=1099511628211ull;}
            identity=std::to_string(hash);
        }
        const auto cache=cfg.cache_dir/identity/
            (cfg.provider=="trt-rtx" ? common::CurrentCudaDeviceCacheKey():"cpu");
        runner=std::make_unique<common::OrtRunner>(common::SharedInferenceEnvironment(),path.string(),cfg.provider,
                    cache.string(),common::EpContextOptions{(cache/"context").string()},profile);
        binding=std::make_unique<Ort::IoBinding>(runner->session);
        if(cfg.provider=="trt-rtx") options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders,"1");
        Ort::AllocatorWithDefaultOptions allocator;
        json metadata;std::ifstream metadata_file(cfg.model_dir/"metadata.json");metadata_file>>metadata;
        for(size_t i=0;i<runner->session.GetOutputCount();++i) {
            const auto key=runner->session.GetOutputNameAllocated(i,allocator);
            const auto type_info=runner->session.GetOutputTypeInfo(i);
            const auto info=type_info.GetTensorTypeAndShapeInfo();
            auto shape=info.GetShape();
            if(std::any_of(shape.begin(),shape.end(),[](int64_t d){return d<=0;}) && metadata.at("graphs").is_object())
                for(const auto& output:metadata.at("graphs").at(graph).at("outputs"))
                    if(output.at("name")==key.get()) shape=output.at("shape").get<std::vector<int64_t>>();
            if(std::any_of(shape.begin(),shape.end(),[](int64_t d){return d<=0;}))
                throw std::runtime_error("SAM requires static output storage for "+graph+"/"+key.get());
            auto output=std::make_unique<Tensor>(shape,info.GetElementType(),cfg.provider=="trt-rtx",cfg.device);
            binding->BindOutput(key.get(),output->value);
            outputs.emplace(key.get(),std::move(output));
        }
    }
    ~Graph() {DeviceScope scope(gpu,device);if(replay) cudaGraphExecDestroy(replay);outputs.clear();binding.reset();runner.reset();}
    Tensor& Out(std::string_view key) {
        const auto it=outputs.find(key);
        if(it==outputs.end()) throw std::runtime_error("Missing SAM graph output "+name+"/"+std::string(key));
        return *it->second;
    }
    void Bind(const char* key,Tensor& tensor) {binding->BindInput(key,tensor.value);}
    void Run() {
        if(replay) {Check(cudaGraphLaunch(replay,stream));return;}
        try {runner->session.Run(options,*binding);}
        catch(const std::exception& e) {throw std::runtime_error("SAM "+name+": "+e.what());}
    }
    void Capture() {
        // Capturing the warmed ORT enqueue also removes ORT's per-call CPU
        // binding/run bookkeeping from the hot path. Tensor addresses stay fixed.
        cudaGraph_t graph=nullptr;
        Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try {Run();}
        catch(...) {cudaStreamEndCapture(stream,&graph);if(graph) cudaGraphDestroy(graph);throw;}
        const auto status=cudaStreamEndCapture(stream,&graph);
        if(status!=cudaSuccess) throw std::runtime_error("SAM capture "+name+": "+cudaGetErrorString(status));
        const auto instantiate=cudaGraphInstantiate(&replay,graph,0);
        cudaGraphDestroy(graph);Check(instantiate);
    }
};
float Sigmoid(float x) {return x>=0 ? 1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));}
float ReadScalar(const void* data,size_t index,bool half) {
    return half ? static_cast<const Ort::Float16_t*>(data)[index].ToFloat():static_cast<const float*>(data)[index];
}
void PutScalar(void* data,size_t index,float value,bool half) {
    if(half) static_cast<Ort::Float16_t*>(data)[index]=Ort::Float16_t(value);
    else static_cast<float*>(data)[index]=value;
}
float Sample(const float* data,int size,int x,int y,int width,int height) {
    const float fx=std::max(0.f,(x+.5f)*size/width-.5f),fy=std::max(0.f,(y+.5f)*size/height-.5f);
    const int x0=std::min(int(fx),size-1),x1=std::min(x0+1,size-1),y0=std::min(int(fy),size-1),y1=std::min(y0+1,size-1);
    const float a=data[y0*size+x0]*(1-(fx-x0))+data[y0*size+x1]*(fx-x0);
    const float b=data[y1*size+x0]*(1-(fx-x0))+data[y1*size+x1]*(fx-x0);
    return a*(1-(fy-y0))+b*(fy-y0);
}
void HostOutput(const float* masks,const float* valid,const float* scores,const float* logits,const float* boxes,int objects,int size,const MaskOutputView& out) {
    const int bytes=out.type==ScalarType::Float16 ? 2:out.type==ScalarType::UInt8 ? 1:4;
    const auto row=out.row_bytes ? out.row_bytes:ptrdiff_t(out.width)*bytes;
    const auto plane=out.instance_bytes ? out.instance_bytes:row*out.height;
    const int planes=out.format==MaskFormat::Labels || out.selected_id ? 1:objects;
    for(int o=0;o<planes;++o) for(int y=0;y<out.height;++y) for(int x=0;x<out.width;++x) {
        char* target=static_cast<char*>(out.data)+o*plane+y*row+x*bytes;
        const int object=out.selected_id ? out.selected_id-1:o;
        if(out.format==MaskFormat::Labels) {
            float best=0;uint32_t id=0;
            for(int i=0;i<objects;++i) if(valid[i]>0 && (!out.selected_id || i==object)) {
                const float p=Sigmoid(Sample(masks+size_t(i)*size*size,size,x,y,out.width,out.height));
                if(p>.5f && p>best) {best=p;id=i+1;}
            }
            *reinterpret_cast<uint32_t*>(target)=id;
        } else {
            float v=Sample(masks+size_t(object)*size*size,size,x,y,out.width,out.height);
            if(out.format!=MaskFormat::Logits) v=valid[object]>0 ? (out.format==MaskFormat::Probabilities ? Sigmoid(v):float(v>0)):0;
            if(out.type==ScalarType::Float16) *reinterpret_cast<Ort::Float16_t*>(target)=Ort::Float16_t(v);
            else if(out.type==ScalarType::UInt8) *reinterpret_cast<uint8_t*>(target)=uint8_t(v);
            else *reinterpret_cast<float*>(target)=v;
        }
    }
    for(int i=0;i<objects;++i) {
        if(out.validity) out.validity[i]=valid[i];
        if(out.scores) out.scores[i]=scores[i];
        if(out.ids) out.ids[i]=valid[i]>0 ? i+1:0;
        if(out.presence_logits) out.presence_logits[i]=logits[i];
        if(out.detection_boxes) std::copy_n(boxes+i*4,4,out.detection_boxes+i*4);
    }
}
int HistorySlot(const TemporalMeta& m,int i) {return i==0 ? 0:((m.next_slot-1-(m.history_count-1)+(i-1)+15)%15)+1;}
} // namespace

struct PreparedPrompt::Impl {
    const void* owner;
    std::map<std::string,std::unique_ptr<Tensor>,std::less<>> values;
};
struct State::Impl {
    const void* owner=nullptr;
    uint64_t generation=0;
    bool leased=false,ready=false,tracking=true,confirmed=false,pending=false;
    size_t step=0,length=1,active=0,next=0;
    int width=0,height=0;
    PreparedPrompt prompt;
    std::unique_ptr<Buffer> ring,meta,status;
    cudaEvent_t status_event=nullptr;
    bool gpu=false;
    int device=0;
    ~Impl() {DeviceScope scope(gpu,device);if(status_event) cudaEventDestroy(status_event);}
};
struct CompletionToken::Slot {
    cudaEvent_t event=nullptr;
    bool leased=false;
    int device=0;
    std::unique_ptr<Buffer> diagnostic;
    ~Slot() {DeviceScope scope(event!=nullptr,device);if(event) cudaEventDestroy(event);}
};
struct Sam::Impl {
    SamConfig cfg;
    json metadata;
    bool gpu=false,half=false;
    cudaStream_t stream=nullptr;
    OwnedStream owned_stream;
    int objects=0,size=0,max_points=0;
    int frame_width=0,frame_height=0;
    size_t state_bytes=0;
    std::map<std::string,std::unique_ptr<Graph>,std::less<>> graphs;
    std::map<std::string,std::unique_ptr<Tensor>,std::less<>> tensors;
    std::vector<std::unique_ptr<State>> states;
    std::vector<std::unique_ptr<CompletionToken::Slot>> events;
    std::unique_ptr<sam3::TextTokenizer> tokenizer;
    std::unique_ptr<Buffer> host_source,host_masks,host_valid,host_scores,host_logits,host_boxes;
    std::vector<std::shared_ptr<PreparedPrompt::Impl>> prompts;
    PreparedPrompt::Impl* current_prompt=nullptr;

    explicit Impl(const SamConfig& config):cfg(config) {
        if(cfg.provider!="trt-rtx" && cfg.provider!="cpu") throw std::invalid_argument("SAM provider must be trt-rtx or cpu");
        gpu=cfg.provider=="trt-rtx";
        if(cfg.max_source_width<=0 || cfg.max_source_height<=0 || cfg.max_source_width>8192 || cfg.max_source_height>8192 ||
           !cfg.completion_capacity || !cfg.state_capacity ||
           !std::isfinite(cfg.confidence_threshold) || cfg.confidence_threshold<0 || cfg.confidence_threshold>1)
            throw std::invalid_argument("Invalid SAM capacities or confidence");
        DeviceScope scope(gpu,cfg.device);
        std::ifstream input(cfg.model_dir/"metadata.json");input>>metadata;
        if(!metadata.contains("attention_contract") || metadata.at("attention_contract").at("version")!=1 ||
           metadata.at("attention_contract").at("spatial_frames")!=7)
            throw std::runtime_error("SAM requires fixed-attention contract v1; prepare a static_state_v2 bundle");
        if(cfg.model==ModelFamily::Sam3 && metadata.value("memory_boundary_version",0)!=2)
            throw std::runtime_error("SAM3 requires corrected memory boundary v2");
        if(cfg.model==ModelFamily::Sam2 && (metadata.value("image_size",0)!=1024 || metadata.value("max_memory_frames",0)!=7 ||
            metadata.value("hidden_dim",0)!=256 || metadata.value("mem_dim",0)!=64 ||
            (metadata.value("dtype",std::string{})!="float16" && metadata.value("dtype",std::string{})!="float32")))
            throw std::runtime_error("Unsupported SAM2 spatial/state/precision contract");
        if(cfg.model==ModelFamily::Sam3 && (metadata.value("model_family",std::string{})!="sam3.1" ||
            metadata.value("image_size",0)!=1008 || metadata.value("max_objects",0)!=16 ||
            metadata.value("dtype",std::string{})!="float32" || metadata.at("attention_contract").value("pointer_frames",0)!=16))
            throw std::runtime_error("Unsupported SAM3 spatial/state/precision contract");
        if(gpu) {
            common::RegisterTensorRTRTXProvider(common::SharedInferenceEnvironment());
            stream=static_cast<cudaStream_t>(cfg.stream);
            if(!stream) {Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));owned_stream.value=stream;owned_stream.device=cfg.device;}
        }
        objects=cfg.model==ModelFamily::Sam2 ? 1:16;
        size=cfg.model==ModelFamily::Sam2 ? 256:288;
        half=cfg.model==ModelFamily::Sam2 && metadata.value("dtype",std::string{})=="float16";
        max_points=cfg.model==ModelFamily::Sam2 ? metadata.at("max_points").get<int>():0;
        // Recurrent storage only, excluding graph outputs and workspace:
        // SAM2: two 7x4096x64 FP16 rings = 7 MiB (FP32 reference = 14 MiB).
        // SAM3: 16 x (4x256x72x72 + 16x256) FP32 = 324.25 MiB.
        // Keep the full history and precision; no compact/shared-history scheme.
        state_bytes=cfg.model==ModelFamily::Sam2 ? 2*7*4096*64*(half?2:4):16*Sam3Entry*sizeof(float);
        Build();
        if(!cfg.diagnostic_dir.empty()) std::filesystem::create_directories(cfg.diagnostic_dir);
        for(size_t i=0;i<cfg.completion_capacity;++i) {
            auto event=std::make_unique<CompletionToken::Slot>();event->device=cfg.device;
            if(gpu) {
                Check(cudaEventCreateWithFlags(&event->event,cudaEventDisableTiming));
                event->diagnostic=std::make_unique<Buffer>(sizeof(int),false,cfg.device,true);
            }
            events.push_back(std::move(event));
        }
        if(gpu) {
            // Pay lazy EP initialization and capture costs at construction.
            for(int pass=0;pass<2;++pass) {
                G("vision").Run();
                if(objects==16) {G("text").Run();G("detector").Run();G("seed").Run();G("seed_memory").Run();}
                else G("prompt").Run();
                G("attention").Run();G("track").Run();G("memory").Run();
            }
            Check(cudaStreamSynchronize(stream));
            if(cfg.cuda_graph) {
                for(auto& [name,graph]:graphs) graph->Capture();
                Check(cudaStreamSynchronize(stream));
            }
        }
    }
    ~Impl() {
        DeviceScope scope(gpu,cfg.device);
        if(gpu) cudaStreamSynchronize(stream);
        states.clear();events.clear();prompts.clear();tensors.clear();graphs.clear();
        host_source.reset();host_masks.reset();host_valid.reset();host_scores.reset();
    }
    Graph& G(std::string_view key) {return *graphs.find(key)->second;}
    Tensor& T(std::string_view key) {return *tensors.find(key)->second;}
    Tensor& New(const char* key,std::vector<int64_t> shape,ONNXTensorElementDataType type=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,size_t reserved_bytes=0) {
        auto tensor=std::make_unique<Tensor>(std::move(shape),type,gpu,cfg.device,reserved_bytes);
        if(gpu) Check(cudaMemsetAsync(tensor->Data(),0,tensor->storage->bytes,stream));
        else std::memset(tensor->Data(),0,tensor->storage->bytes);
        auto& result=*tensor;tensors.emplace(key,std::move(tensor));return result;
    }
    void Copy(void* dst,const void* src,size_t bytes) {
        if(gpu) Check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));else std::memcpy(dst,src,bytes);
    }
    void Upload(void* dst,const void* src,size_t bytes) {
        // Setup only. New() queues zeroing on the nonblocking bound stream;
        // default-stream cudaMemcpy would race that zeroing and could erase
        // temporal embeddings or scalar constants after their upload.
        if(gpu) {Check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream));Check(cudaStreamSynchronize(stream));}
        else std::memcpy(dst,src,bytes);
    }
    void Scalar(Tensor& dst,float value) {
        std::array<uint8_t,4> data{};PutScalar(data.data(),0,value,half);
        Upload(dst.Data(),data.data(),dst.storage->bytes);
    }
    void AddGraph(const char* key,const char* name) {
        try {graphs.emplace(key,std::make_unique<Graph>(cfg,name,stream));}
        catch(const std::exception& e) {throw std::runtime_error("SAM graph initialization "+std::string(name)+": "+e.what());}
    }
    void Build();
    void Encode(const FrameView&);
    void Detect(bool tracking);
    void Seed();
    void Pack(State::Impl&);
    void Commit(State::Impl&);
    void Run2(State::Impl&,bool initialize);
    void Run3(State::Impl&);
    void Validate(const FrameView&,const MaskOutputView&,State::Impl&,const State*);
    void WarmKernels(State::Impl&);
    void Dump(State::Impl&);
};

void Sam::Impl::Build() {
    const int image_size=objects==1 ? 1024:1008;
    const auto scalar=half ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    New("rgb",{cfg.max_source_height,cfg.max_source_width,3},ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    New("image",{1,3,image_size,image_size},scalar);
    New("masks",{objects,1,size,size});New("valid",{objects});New("scores",{objects});New("presence",{objects});New("boxes",{objects,4});
    New("memory_valid",{7},ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    host_source=std::make_unique<Buffer>(size_t(cfg.max_source_width)*cfg.max_source_height*3,false,cfg.device,gpu);
    host_masks=std::make_unique<Buffer>(size_t(objects)*size*size*4,false,cfg.device,gpu);
    host_valid=std::make_unique<Buffer>(objects*4,false,cfg.device,gpu);
    host_scores=std::make_unique<Buffer>(objects*4,false,cfg.device,gpu);
    host_logits=std::make_unique<Buffer>(objects*4,false,cfg.device,gpu);
    host_boxes=std::make_unique<Buffer>(objects*16,false,cfg.device,gpu);
    AddGraph("vision",objects==1 ? "image_encoder":"vision_encoder");
    G("vision").Bind("image",T("image"));
    if(!gpu) AddGraph("preprocess","image_preprocess");
    if(objects==1) {
        New("coords",{1,max_points,2},scalar);New("labels",{1,max_points},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        New("empty_coords",{1,1,2},scalar);New("empty_labels",{1,1},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        const int32_t empty=-1;Upload(T("empty_labels").Data(),&empty,sizeof(empty));
        New("input_masks",{1,1,256,256},scalar);New("has_masks",{1,1,1,1},scalar);
        New("from_points",{},scalar);New("one",{},scalar);New("zero",{},scalar);
        Scalar(T("one"),1);Scalar(T("zero"),0);
        New("memory",{7*4096,1,64},scalar);New("memory_pos",{7*4096,1,64},scalar);
        AddGraph("prompt","sam2_decoder");AddGraph("track","sam2_decoder_propagate");
        AddGraph("attention","memory_attention");AddGraph("memory","memory_encoder");
        for(auto key:{"prompt","track"}) {
            G(key).Bind("image_features_0",G("vision").Out("image_features_0"));
            G(key).Bind("image_features_1",G("vision").Out("image_features_1"));
            G(key).Bind("image_embeddings",key==std::string_view("prompt") ? G("vision").Out("image_embeddings"):G("attention").Out("conditioned_image_embeddings"));
            G(key).Bind("point_coords",T(key==std::string_view("prompt") ? "coords":"empty_coords"));
            G(key).Bind("point_labels",T(key==std::string_view("prompt") ? "labels":"empty_labels"));
            G(key).Bind("input_masks",T("input_masks"));G(key).Bind("has_input_masks",T("has_masks"));
        }
        G("attention").Bind("vision_feat",G("vision").Out("vision_feat"));G("attention").Bind("vision_pos",G("vision").Out("vision_pos"));
        G("attention").Bind("memory",T("memory"));G("attention").Bind("memory_pos",T("memory_pos"));G("attention").Bind("memory_valid",T("memory_valid"));
        G("memory").Bind("vision_feat",G("vision").Out("vision_feat"));
        // A stable intermediate avoids rebinding memory inputs between decoders.
        New("high_masks",{1,1,1024,1024},scalar);New("object_logits",{1,1},scalar);
        G("memory").Bind("pred_masks_high_res",T("high_masks"));G("memory").Bind("object_score_logits",T("object_logits"));
        G("memory").Bind("is_mask_from_pts",T("from_points"));
    } else {
        New("resize_scratch",{cfg.max_source_height,1008,3});
        New("tokens",{1,32},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        New("candidate_valid",{16});New("propagation_valid",{16});New("zero_valid",{16});
        New("selected",{16},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);New("error",{1},ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        New("packed",{16,1,288,288});New("seeds",{16,1,1152,1152});
        // Workspace, not recurrent state: reserve 81 MiB for either corrected
        // seed-memory boundary, but bind/capture the ordinary 1008-square view.
        New("consolidated",{16,1,1008,1008},ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,16*1152*1152*sizeof(float));
        New("pointer_valid",{16},ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        for(auto key:{"mi","mip","memory","memory_pos"}) New(key,{7*5184,1,256});
        New("pointers",{256,1,256});New("pointer_positions",{256,1,256});
        New("decision",{int64_t(sizeof(detail::Decision))},ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        AddGraph("text","text_encoder");AddGraph("detector","concept_detector");
        AddGraph("seed","seed_decoder");AddGraph("seed_memory","seed_memory");
        AddGraph("attention","memory_attention");AddGraph("track","tracking_decoder");AddGraph("memory","tracking_memory");
        G("text").Bind("token_ids",T("tokens"));
        for(auto& [name,tensor]:G("text").outputs) {
            auto& target=New(name.c_str(),tensor->shape,tensor->type);
            G("detector").Bind(name.c_str(),target);
        }
        for(auto& [name,tensor]:G("vision").outputs) if(name.starts_with("detector_") && name!="detector_position_0" && name!="detector_position_1") G("detector").Bind(name.c_str(),*tensor);
        for(int i=0;i<3;++i) {
            const auto feature="feature_"+std::to_string(i);
            G("seed").Bind(feature.c_str(),G("vision").Out("interactive_"+feature));
            if(i<2) G("track").Bind(feature.c_str(),G("vision").Out("propagation_"+feature));
        }
        G("seed").Bind("masks",T("seeds"));G("seed").Bind("valid",T("candidate_valid"));
        for(auto key:{"seed_memory","memory"}) {
            G(key).Bind("image_features",G("vision").Out("propagation_feature_2"));
            G(key).Bind("mask_logits",key==std::string_view("seed_memory") ? T("consolidated"):G("track").Out("mask_logits"));
            G(key).Bind("object_logits",G(key==std::string_view("seed_memory") ? "seed":"track").Out("object_logits"));
            G(key).Bind("valid",T(key==std::string_view("seed_memory") ? "candidate_valid":"propagation_valid"));
            G(key).Bind("conditioning",T(key==std::string_view("seed_memory") ? "candidate_valid":"zero_valid"));
        }
        G("attention").Bind("image_features",G("vision").Out("propagation_feature_2"));G("attention").Bind("image_position",G("vision").Out("propagation_position_2"));
        G("attention").Bind("memory_image",T("mi"));G("attention").Bind("memory_image_position",T("mip"));
        G("attention").Bind("memory",T("memory"));G("attention").Bind("memory_position",T("memory_pos"));
        G("attention").Bind("object_pointers",T("pointers"));G("attention").Bind("pointer_position",T("pointer_positions"));
        G("attention").Bind("memory_valid",T("memory_valid"));G("attention").Bind("pointer_valid",T("pointer_valid"));
        G("track").Bind("conditioned",G("attention").Out("conditioned"));G("track").Bind("valid",T("propagation_valid"));
        tokenizer=std::make_unique<sam3::TextTokenizer>(cfg.model_dir/"tokenizer.json");
        json constants;std::ifstream f(cfg.model_dir/"temporal_constants.json");f>>constants;
        std::vector<float> embeddings;
        for(const auto& row:constants.at("mask_memory_temporal_embeddings")) {auto v=row.get<std::vector<float>>();embeddings.insert(embeddings.end(),v.begin(),v.end());}
        if(embeddings.size()!=7*256) throw std::runtime_error("Invalid SAM3 temporal embedding size");
        Upload(New("embeddings",{7,256}).Data(),embeddings.data(),embeddings.size()*4);
        if(constants.at("pointer_projection")=="Linear") {
            std::vector<float> weights;for(const auto& row:constants.at("pointer_projection_weight")) {auto v=row.get<std::vector<float>>();weights.insert(weights.end(),v.begin(),v.end());}
            if(weights.size()!=256*256) throw std::runtime_error("Invalid SAM3 pointer projection");
            auto bias=constants.at("pointer_projection_bias").is_null() ? std::vector<float>(256,0):constants.at("pointer_projection_bias").get<std::vector<float>>();
            Upload(New("weights",{256,256}).Data(),weights.data(),weights.size()*4);Upload(New("bias",{256}).Data(),bias.data(),bias.size()*4);
        }
    }
}

void Sam::Impl::Encode(const FrameView& frame) {
    frame_width=frame.width;frame_height=frame.height;
    if(gpu && frame.memory==MemoryLocation::Device) detail::ConvertFrame(frame,T("rgb").As<uint8_t>(),stream);
    else {
        if(gpu) Check(cudaStreamSynchronize(stream)); // Explicit host-input staging boundary.
        auto* rgb=gpu ? host_source->As<uint8_t>():T("rgb").As<uint8_t>();
        for(int y=0;y<frame.height;++y) {
            const int sy=frame.origin==Origin::TopLeft ? y:frame.height-1-y;
            const char* row=static_cast<const char*>(frame.data)+sy*frame.row_bytes;
            if(frame.format==FrameFormat::Rgb8) std::memcpy(rgb+size_t(y)*frame.width*3,row,size_t(frame.width)*3);
            else for(int x=0;x<frame.width;++x) for(int c=0;c<3;++c) {
                const float* pixel=reinterpret_cast<const float*>(row)+4*x;
                float v=pixel[c];if(frame.premultiplied) v=pixel[3]>0 ? v/pixel[3]:0;
                rgb[(size_t(y)*frame.width+x)*3+c]=std::isfinite(v) ? uint8_t(std::round(std::clamp(v,0.f,1.f)*255.f)):0;
            }
        }
        if(gpu) Check(cudaMemcpyAsync(T("rgb").Data(),rgb,size_t(frame.width)*frame.height*3,cudaMemcpyHostToDevice,stream));
    }
    if(gpu) {
        if(objects==1) detail::PreprocessSam2(T("rgb").As<uint8_t>(),frame.width,frame.height,T("image").Data(),half,stream);
        else sam3::PreprocessDeviceRgb(T("rgb").As<uint8_t>(),frame.width,frame.height,ptrdiff_t(frame.width)*3,T("resize_scratch").As(),T("image").As(),stream);
    } else {
        const int64_t shape[]{1,frame.height,frame.width,3};
        const auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto rgb=Ort::Value::CreateTensor(memory,T("rgb").Data(),size_t(frame.width)*frame.height*3,shape,4,ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        G("preprocess").binding->BindInput("image_rgb",rgb);G("preprocess").Run();
        Copy(T("image").Data(),G("preprocess").Out("image").Data(),T("image").storage->bytes);
    }
    G("vision").Run();
}
void Sam::Impl::Detect(bool tracking) {
    G("detector").Run();
    const float threshold=tracking ? std::max(.65f,cfg.confidence_threshold):cfg.confidence_threshold;
    if(gpu) {
        Check(cudaMemsetAsync(T("error").Data(),0,sizeof(int),stream));
        sam3::SelectQueries(G("detector").Out("class_logits").As(),G("detector").Out("presence_logits").As(),200,threshold,
                            T("selected").As<int>(),T("candidate_valid").As(),T("error").As<int>(),stream);
        sam3::GatherMasks(G("detector").Out("mask_logits").As(),T("selected").As<int>(),T("packed").As(),stream);
        detail::DetectionScores(G("detector").Out("class_logits").As(),G("detector").Out("presence_logits").As(),
                            T("selected").As<int>(),T("candidate_valid").As(),T("scores").As(),T("presence").As(),
                            G("detector").Out("boxes_xyxy").As(),T("boxes").As(),frame_width,frame_height,stream);
    } else {
        auto* selected=T("selected").As<int>();auto* valid=T("candidate_valid").As();
        std::fill_n(selected,16,-1);std::fill_n(valid,16,0.f);std::fill_n(T("scores").As(),16,0.f);
        const auto* classes=G("detector").Out("class_logits").As();
        const float presence=Sigmoid(*G("detector").Out("presence_logits").As());
        if(!std::isfinite(presence)) throw std::runtime_error("SAM3 non-finite presence");
        int count=0;
        for(int i=0;i<200;++i) {
            if(!std::isfinite(classes[i])) throw std::runtime_error("SAM3 non-finite detection score");
            const float score=Sigmoid(classes[i])*presence;
            if(score<=threshold) continue;
            if(count==16) throw std::runtime_error("SAM3 object capacity exceeded (16)");
            selected[count]=i;valid[count]=1;T("scores").As()[count]=score;++count;
        }
        for(int i=0;i<16;++i) {
            const float score=T("scores").As()[i];
            T("presence").As()[i]=score<=0 ? -1024.f:score>=1 ? 1024.f:std::log(score/(1-score));
            for(int k=0;k<4;++k) T("boxes").As()[i*4+k]=selected[i]>=0 ? G("detector").Out("boxes_xyxy").As()[selected[i]*4+k]*(k%2 ? frame_height:frame_width):0;
            auto* output=T("packed").As()+size_t(i)*288*288;
            if(selected[i]<0) std::fill_n(output,288*288,-1024.f);
            else std::copy_n(G("detector").Out("mask_logits").As()+size_t(selected[i])*288*288,288*288,output);
        }
    }
}
void Sam::Impl::Seed() {
    if(gpu) {
        for(int i=0;i<16;++i) sam3::ResizeMaskProbability(T("packed").As()+size_t(i)*288*288,288,288,
                T("seeds").As()+size_t(i)*1152*1152,1152,1152,stream,true);
    } else {
        for(int i=0;i<16;++i) for(int y=0;y<1152;++y) for(int x=0;x<1152;++x)
            T("seeds").As()[(size_t(i)*1152+y)*1152+x]=Sample(T("packed").As()+size_t(i)*288*288,288,x,y,1152,1152)>0 ? 1.f:0.f;
    }
    G("seed").Run();
    if(gpu) sam3::PrepareSeedMemoryMasks(G("seed").Out("mask_logits").As(),T("candidate_valid").As(),T("consolidated").As(),stream);
    else {
        // Native memory boundary v2: bilinear high-resolution logits, then
        // non-overlap suppression. This is deliberately not the binary prompt.
        const auto* low=G("seed").Out("mask_logits").As();const auto* valid=T("candidate_valid").As();
        auto* high=T("consolidated").As();
        for(int y=0;y<1008;++y) for(int x=0;x<1008;++x) {
            float values[16],best=-std::numeric_limits<float>::infinity();int winner=0;
            for(int i=0;i<16;++i) {values[i]=valid[i]>0 ? Sample(low+size_t(i)*288*288,288,x,y,1008,1008):-1024.f;
                if(values[i]>best) {best=values[i];winner=i;}}
            for(int i=0;i<16;++i) high[(size_t(i)*1008+y)*1008+x]=i==winner ? values[i]:std::min(values[i],-10.f);
        }
    }
    G("seed_memory").Run();
}
void Sam::Impl::Pack(State::Impl& state) {
    const auto weight=tensors.find("weights"),bias=tensors.find("bias");
    const float* weights=weight==tensors.end() ? nullptr:weight->second->As();
    const float* biases=bias==tensors.end() ? nullptr:bias->second->As();
    if(gpu) {
        detail::PackSam3(state.meta->As<TemporalMeta>(),state.ring->As(),T("embeddings").As(),weights,biases,
            T("mi").As(),T("mip").As(),T("memory").As(),T("memory_pos").As(),T("pointers").As(),T("pointer_positions").As(),
            T("memory_valid").As<bool>(),T("pointer_valid").As<bool>(),T("propagation_valid").As(),stream);
        return;
    }
    const auto& m=*state.meta->As<TemporalMeta>();
    std::copy_n(m.valid,16,T("propagation_valid").As());
    for(int i=0;i<7;++i) {
        const bool active=i<std::min(m.history_count,7);T("memory_valid").As<bool>()[i]=active;
        const int history=i==0 ? 0:(m.history_count>7 ? m.history_count-7+i:i);
        const int slot=active ? HistorySlot(m,history):0;
        const uint64_t age=active ? m.step-m.frames[slot]:0;
        const int t=age>=7 ? 6:(i==0 ? 6-int(age):int(age)-1);
        const float* base=state.ring->As()+size_t(slot)*Sam3Entry;
        for(size_t p=0;p<5184;++p) for(size_t c=0;c<256;++c) {
            const size_t dst=size_t(i)*Feature+p*256+c,src=c*5184+p;
            const float temporal=active ? T("embeddings").As()[t*256+c]:0;
            T("mi").As()[dst]=active ? base[src]:0;
            T("mip").As()[dst]=active ? base[src+Feature]+temporal:0;
            T("memory").As()[dst]=active ? base[src+2*Feature]:0;
            T("memory_pos").As()[dst]=active ? base[src+3*Feature]+temporal:0;
        }
    }
    for(int i=0;i<16;++i) {
        const bool active=i<m.history_count;T("pointer_valid").As<bool>()[i]=active;
        const int slot=active ? HistorySlot(m,i):0;
        const float distance=active ? float(m.step-m.frames[slot])/std::max<size_t>(1,std::min<size_t>(16,m.length)-1):0;
        for(int c=0;c<256;++c) {
            float projected=active && biases ? biases[c]:0;
            if(active) for(int j=0;j<(weights?256:1);++j) {
                const int k=weights ? j:c;
                const float phase=distance/std::pow(10000.f,2.f*((k%128)/2)/128.f);
                const float sine=k<128 ? std::sin(phase):std::cos(phase);
                projected+=weights ? weights[c*256+j]*sine:sine;
            }
            for(int o=0;o<16;++o) {
                const size_t dst=(size_t(i)*16+o)*256+c;
                T("pointers").As()[dst]=active ? state.ring->As()[size_t(slot)*Sam3Entry+4*Feature+o*256+c]:0;
                T("pointer_positions").As()[dst]=projected;
            }
        }
    }
}
void Sam::Impl::Commit(State::Impl& state) {
    auto& vision=G("vision");auto& seed=G("seed");auto& track=G("track");
    auto& sm=G("seed_memory");auto& tm=G("memory");
    if(gpu) {
        detail::CommitSam3(state.meta->As<TemporalMeta>(),state.ring->As(),T("decision").As<detail::Decision>(),
            vision.Out("propagation_feature_2").As(),vision.Out("propagation_position_2").As(),
            sm.Out("memory_features").As(),sm.Out("memory_position").As(),seed.Out("object_pointers").As(),
            T("candidate_valid").As(),seed.Out("mask_logits").As(),seed.Out("object_logits").As(),
            tm.Out("memory_features").As(),tm.Out("memory_position").As(),track.Out("object_pointers").As(),
            track.Out("mask_logits").As(),track.Out("object_logits").As(),T("masks").As(),T("valid").As(),T("scores").As(),T("presence").As(),stream);
        return;
    }
    auto& m=*state.meta->As<TemporalMeta>();
    const bool initialize=!m.initialized;
    const auto* candidate=T("candidate_valid").As();
    const bool any=std::any_of(candidate,candidate+16,[](float f){return f>0;});
    const bool commit=m.initialized || any;
    auto& decoded=initialize ? seed:track;auto& encoded=initialize ? sm:tm;
    const int slot=initialize ? 0:m.next_slot;
    const auto* valid=initialize ? candidate:m.valid;
    for(int i=0;i<16;++i) {
        const float logit=decoded.Out("object_logits").As()[i];
        T("valid").As()[i]=commit && valid[i]>0 && logit>0 ? 1.f:0.f;
        T("scores").As()[i]=commit && valid[i]>0 ? Sigmoid(logit):0;
        T("presence").As()[i]=commit && valid[i]>0 ? logit:-1024.f;
        auto* destination=T("masks").As()+size_t(i)*288*288;
        if(commit && valid[i]>0) std::copy_n(decoded.Out("mask_logits").As()+size_t(i)*288*288,288*288,destination);
        else std::fill_n(destination,288*288,-1024.f);
    }
    if(commit) {
        auto* destination=state.ring->As()+size_t(slot)*Sam3Entry;
        std::copy_n(vision.Out("propagation_feature_2").As(),Feature,destination);
        std::copy_n(vision.Out("propagation_position_2").As(),Feature,destination+Feature);
        std::copy_n(encoded.Out("memory_features").As(),Feature,destination+2*Feature);
        std::copy_n(encoded.Out("memory_position").As(),Feature,destination+3*Feature);
        for(int i=0;i<16;++i) for(int c=0;c<256;++c) destination[4*Feature+i*256+c]=valid[i]>0 ? decoded.Out("object_pointers").As()[i*256+c]:0;
        m.frames[slot]=m.step;
        if(initialize) {std::copy_n(candidate,16,m.valid);m.initialized=1;m.history_count=1;m.next_slot=1;}
        else {m.history_count=std::min(m.history_count+1,16);m.next_slot=1+m.next_slot%15;}
    }
    ++m.step;
}
void Sam::Impl::Run2(State::Impl& state,bool initialize) {
    const bool seed=initialize || !state.tracking || state.active==0;
    if(!seed) {
        Copy(T("memory").Data(),state.ring->data,state_bytes/2);
        Copy(T("memory_pos").Data(),static_cast<char*>(state.ring->data)+state_bytes/2,state_bytes/2);
        if(gpu) detail::MemoryValidity(T("memory_valid").As<bool>(),int(state.active),stream);
        else for(int i=0;i<7;++i) T("memory_valid").As<bool>()[i]=size_t(i)<state.active;
        G("attention").Run();
    }
    auto& decoder=G(seed ? "prompt":"track");decoder.Run();
    if(state.tracking) {
        Copy(T("high_masks").Data(),decoder.Out("masks").Data(),T("high_masks").storage->bytes);
        Copy(T("object_logits").Data(),decoder.Out("object_score_logits").Data(),T("object_logits").storage->bytes);
        Copy(T("from_points").Data(),T(seed ? "one":"zero").Data(),T("from_points").storage->bytes);
        G("memory").Run();
        const size_t bytes=4096*64*(half?2:4);
        Copy(static_cast<char*>(state.ring->data)+state.next*bytes,G("memory").Out("memory_features").Data(),bytes);
        Copy(static_cast<char*>(state.ring->data)+state_bytes/2+state.next*bytes,G("memory").Out("memory_pos").Data(),bytes);
        state.active=std::min(state.active+1,size_t(7));state.next=(state.next+1)%7;
    }
    if(gpu) detail::Sam2Result(decoder.Out("low_res_masks").Data(),decoder.Out("object_score_logits").Data(),half,
                                T("masks").As(),T("valid").As(),T("scores").As(),T("presence").As(),stream);
    else {
        for(size_t i=0;i<256*256;++i) T("masks").As()[i]=ReadScalar(decoder.Out("low_res_masks").Data(),i,half);
        *T("valid").As()=1;*T("scores").As()=Sigmoid(ReadScalar(decoder.Out("object_score_logits").Data(),0,half));
        *T("presence").As()=ReadScalar(decoder.Out("object_score_logits").Data(),0,half);
    }
}
void Sam::Impl::Run3(State::Impl& state) {
    if(!state.tracking) {
        Detect(false);Copy(T("masks").Data(),T("packed").Data(),T("masks").storage->bytes);
        Copy(T("valid").Data(),T("candidate_valid").Data(),T("valid").storage->bytes);
        return;
    }
    if(gpu && !state.confirmed && state.pending) {
        const auto status=cudaEventQuery(state.status_event);
        if(status==cudaSuccess) {state.confirmed=state.status->As<TemporalMeta>()->initialized!=0;state.pending=false;}
        else if(status!=cudaErrorNotReady) Check(status);
    }
    if(!gpu) state.confirmed=state.meta->As<TemporalMeta>()->initialized!=0;
    if(!state.confirmed) {Detect(true);Seed();}
    Pack(state);G("attention").Run();G("track").Run();G("memory").Run();Commit(state);
    if(gpu && !state.confirmed && !state.pending) {
        Check(cudaMemcpyAsync(state.status->data,state.meta->data,sizeof(TemporalMeta),cudaMemcpyDeviceToHost,stream));
        Check(cudaEventRecord(state.status_event,stream));state.pending=true;
    }
}

void Sam::Impl::Validate(const FrameView& frame,const MaskOutputView& output,State::Impl& state,const State* previous) {
    if(state.owner!=this || !state.leased) throw std::invalid_argument("Destination state is foreign or released");
    if(previous) {
        const auto& src=*previous->impl_;
        if(src.owner!=this || !src.leased || !src.ready) throw std::invalid_argument("Previous state is foreign, released or uninitialized");
        if(src.width!=frame.width || src.height!=frame.height) throw std::invalid_argument("SAM sequence geometry changed; initialize a new state");
        if(src.step>=src.length && src.tracking) throw std::invalid_argument("SAM sequence exhausted");
    }
    const int pixel=frame.format==FrameFormat::Rgb8 ? 3:16;
    if(!frame.data || frame.width<=0 || frame.height<=0 || frame.width>cfg.max_source_width || frame.height>cfg.max_source_height ||
       frame.row_bytes==PTRDIFF_MIN || std::abs(frame.row_bytes)<ptrdiff_t(frame.width)*pixel ||
       (frame.format==FrameFormat::Rgba32F && frame.row_bytes%4))
        throw std::invalid_argument("Invalid SAM source geometry/stride");
    if((!gpu && frame.memory==MemoryLocation::Device) || (frame.memory==MemoryLocation::Device && frame.device!=cfg.device))
        throw std::invalid_argument("SAM input memory/device mismatch");
    const int bytes=output.type==ScalarType::Float16 ? 2:output.type==ScalarType::UInt8 ? 1:4;
    const ptrdiff_t row=output.row_bytes ? output.row_bytes:ptrdiff_t(output.width)*bytes;
    if(!output.data || output.width<=0 || output.height<=0 || output.width>8192 || output.height>8192 ||
       row<ptrdiff_t(output.width)*bytes || row%bytes || output.instance_bytes<0 || output.instance_bytes%bytes ||
       (output.instance_bytes && output.instance_bytes<row*output.height) || output.selected_id<0 || output.selected_id>objects)
        throw std::invalid_argument("Invalid SAM mask geometry/stride/object selection");
    if((output.format==MaskFormat::Labels)!=(output.type==ScalarType::UInt32) ||
       (output.type==ScalarType::UInt8 && output.format!=MaskFormat::Binary))
        throw std::invalid_argument("Mask labels require UInt32; UInt8 requires binary output");
    if((!gpu && output.memory==MemoryLocation::Device) || (output.memory==MemoryLocation::Device && output.device!=cfg.device))
        throw std::invalid_argument("SAM output memory/device mismatch");
    if(gpu) {
        auto check=[&](const void* data) {
            cudaPointerAttributes attributes{};Check(cudaPointerGetAttributes(&attributes,data));
            if(attributes.type!=cudaMemoryTypeDevice || attributes.device!=cfg.device) throw std::invalid_argument("SAM borrowed allocation is on a different CUDA device");
        };
        if(frame.memory==MemoryLocation::Device) check(frame.data);
        if(output.memory==MemoryLocation::Device) {
            check(output.data);if(output.validity) check(output.validity);if(output.scores) check(output.scores);if(output.ids) check(output.ids);if(output.presence_logits) check(output.presence_logits);if(output.detection_boxes) check(output.detection_boxes);
        }
    }
}

void Sam::Impl::WarmKernels(State::Impl& state) {
    if(!gpu) return;
    DeviceScope scope(true,cfg.device);
    FrameView source{T("rgb").Data(),1,1,3,MemoryLocation::Device};source.device=cfg.device;
    detail::ConvertFrame(source,T("rgb").As<uint8_t>(),stream);
    detail::ResetMeta(state.meta->As<TemporalMeta>(),1,stream);
    if(objects==1) {
        detail::PreprocessSam2(T("rgb").As<uint8_t>(),1,1,T("image").Data(),half,stream);
        detail::MemoryValidity(T("memory_valid").As<bool>(),0,stream);
        detail::Sam2Result(G("prompt").Out("low_res_masks").Data(),G("prompt").Out("object_score_logits").Data(),half,
            T("masks").As(),T("valid").As(),T("scores").As(),T("presence").As(),stream);
    } else {
        sam3::PreprocessDeviceRgb(T("rgb").As<uint8_t>(),1,1,3,T("resize_scratch").As(),T("image").As(),stream);
        sam3::SelectQueries(G("detector").Out("class_logits").As(),G("detector").Out("presence_logits").As(),200,1.f,
            T("selected").As<int>(),T("candidate_valid").As(),T("error").As<int>(),stream);
        sam3::GatherMasks(G("detector").Out("mask_logits").As(),T("selected").As<int>(),T("packed").As(),stream);
        sam3::ResizeMaskProbability(T("packed").As(),288,288,T("seeds").As(),1152,1152,stream,true);
        sam3::PrepareSeedMemoryMasks(G("seed").Out("mask_logits").As(),T("candidate_valid").As(),T("consolidated").As(),stream);
        detail::DetectionScores(G("detector").Out("class_logits").As(),G("detector").Out("presence_logits").As(),T("selected").As<int>(),
            T("candidate_valid").As(),T("scores").As(),T("presence").As(),G("detector").Out("boxes_xyxy").As(),T("boxes").As(),1,1,stream);
        Pack(state);Commit(state);Check(cudaMemsetAsync(T("error").Data(),0,sizeof(int),stream));
    }
    MaskOutputView output{T("masks").Data(),size,size};output.device=cfg.device;
    detail::WriteMasks(T("masks").As(),T("valid").As(),T("scores").As(),T("presence").As(),T("boxes").As(),objects,size,output,stream);
    Check(cudaEventRecord(state.status_event,stream));
    for(auto& event:events) Check(cudaEventRecord(event->event,stream));
    Check(cudaStreamSynchronize(stream));
}
void Sam::Impl::Dump(State::Impl& state) {
    if(cfg.diagnostic_dir.empty() || objects!=16 || !state.tracking) return;
    TemporalMeta meta;
    if(gpu) Check(cudaMemcpy(&meta,state.meta->data,sizeof(meta),cudaMemcpyDeviceToHost));
    else meta=*state.meta->As<TemporalMeta>();
    auto read=[&](Tensor& tensor) {
        std::vector<uint8_t> storage(tensor.storage->bytes);
        if(gpu) Check(cudaMemcpy(storage.data(),tensor.Data(),storage.size(),cudaMemcpyDeviceToHost));
        else std::memcpy(storage.data(),tensor.Data(),storage.size());
        const auto count=storage.size()/TypeBytes(tensor.type);std::vector<float> values(count);
        for(size_t i=0;i<count;++i) {
            if(tensor.type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) values[i]=reinterpret_cast<const float*>(storage.data())[i];
            else if(tensor.type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) values[i]=reinterpret_cast<const Ort::Float16_t*>(storage.data())[i].ToFloat();
            else if(tensor.type==ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) values[i]=storage[i] ? 1.f:0.f;
            else if(tensor.type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) values[i]=float(reinterpret_cast<const int32_t*>(storage.data())[i]);
            else throw std::runtime_error("Unsupported SAM diagnostic tensor dtype");
        }
        return values;
    };
    json record{{"frame_index",state.step},{"object_logits",read(T("presence"))}};
    const auto graph=meta.initialized && meta.frames[0]+1==meta.step ? "seed":"track";
    for(auto& [name,tensor]:G(graph).outputs) if(name.starts_with("debug_")) record[name]=read(*tensor);
    std::ofstream file(cfg.diagnostic_dir/(std::to_string(state.step)+".json"));
    file.exceptions(std::ios::failbit|std::ios::badbit);file<<record.dump(2);
}

Sam::Sam(const SamConfig& config):impl_(std::make_unique<Impl>(config)) {
    ReserveStates(config.state_capacity);
    if(!impl_->states.empty()) impl_->WarmKernels(*impl_->states.front()->impl_);
}
Sam::~Sam()=default;
State::State(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
State::~State()=default;
size_t State::Step() const {return impl_->step;}
StateLease::~StateLease() {
    if(state_ && state_->impl_->generation==generation_) state_->impl_->leased=false;
}
StateLease::StateLease(StateLease&& rhs) noexcept:state_(std::exchange(rhs.state_,nullptr)),generation_(rhs.generation_) {}
StateLease& StateLease::operator=(StateLease&& rhs) noexcept {
    if(this!=&rhs) {
        if(state_ && state_->impl_->generation==generation_) state_->impl_->leased=false;
        state_=std::exchange(rhs.state_,nullptr);generation_=rhs.generation_;
    }
    return *this;
}
CompletionToken::~CompletionToken() {if(slot_) slot_->leased=false;}
CompletionToken::CompletionToken(CompletionToken&& rhs) noexcept:slot_(std::exchange(rhs.slot_,nullptr)) {}
CompletionToken& CompletionToken::operator=(CompletionToken&& rhs) noexcept {
    if(this!=&rhs) {if(slot_) slot_->leased=false;slot_=std::exchange(rhs.slot_,nullptr);}return *this;
}
bool CompletionToken::Ready() const {
    if(!slot_ || !slot_->event) return true;
    DeviceScope scope(true,slot_->device);
    const auto result=cudaEventQuery(slot_->event);
    if(result==cudaErrorNotReady) return false;
    Check(result);
    const int error=*slot_->diagnostic->As<int>();
    if(error) throw std::runtime_error(error==1 ? "SAM3 object capacity exceeded (16)":"SAM3 non-finite detection score");
    return true;
}
void CompletionToken::Wait() const {
    if(slot_ && slot_->event) {DeviceScope scope(true,slot_->device);Check(cudaEventSynchronize(slot_->event));Ready();}
}
void CompletionToken::WaitOn(void* stream) const {
    if(slot_ && slot_->event) {DeviceScope scope(true,slot_->device);Check(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),slot_->event,0));}
}
int Sam::MaxObjects() const {return impl_->objects;}
int Sam::MaskSize() const {return impl_->size;}
size_t Sam::StateBytes() const {return impl_->state_bytes;}
void* Sam::Stream() const {return impl_->stream;}

void Sam::ReserveStates(size_t capacity) {
    auto& p=*impl_;DeviceScope scope(p.gpu,p.cfg.device);
    if(capacity<=p.states.size()) return;
    p.states.reserve(capacity);
    while(p.states.size()<capacity) {
        auto data=std::make_unique<State::Impl>();data->owner=&p;data->gpu=p.gpu;data->device=p.cfg.device;
        data->ring=std::make_unique<Buffer>(p.state_bytes,p.gpu,p.cfg.device);
        data->meta=std::make_unique<Buffer>(sizeof(TemporalMeta),p.gpu,p.cfg.device);
        data->status=std::make_unique<Buffer>(sizeof(TemporalMeta),false,p.cfg.device,p.gpu);
        if(p.gpu) {
            Check(cudaMemsetAsync(data->ring->data,0,p.state_bytes,p.stream));
            Check(cudaMemsetAsync(data->meta->data,0,sizeof(TemporalMeta),p.stream));
            Check(cudaEventCreateWithFlags(&data->status_event,cudaEventDisableTiming));
        } else {std::memset(data->ring->data,0,p.state_bytes);std::memset(data->meta->data,0,sizeof(TemporalMeta));}
        p.states.emplace_back(new State(std::move(data)));
    }
}
StateLease Sam::AcquireState() {
    for(auto& state:impl_->states) if(!state->impl_->leased) {
        auto& s=*state->impl_;s.leased=true;s.ready=false;++s.generation;
        StateLease lease;lease.state_=state.get();lease.generation_=s.generation;return lease;
    }
    throw std::runtime_error("SAM state capacity exhausted; call ReserveStates during setup");
}
void Sam::ReleaseState(StateLease&& lease) {
    if(!lease.state_ || lease.state_->impl_->owner!=impl_.get() || lease.generation_!=lease.state_->impl_->generation)
        throw std::invalid_argument("Foreign or released SAM state lease");
    lease.state_->impl_->leased=false;lease.state_=nullptr;
}
PreparedPrompt Sam::PreparePrompt(const Prompt& prompt) {
    auto& p=*impl_;DeviceScope scope(p.gpu,p.cfg.device);
    if(p.gpu) Check(cudaStreamSynchronize(p.stream)); // Explicit setup boundary.
    auto prepared=std::make_shared<PreparedPrompt::Impl>();prepared->owner=&p;
    if(p.objects==1) {
        if(prompt.labels.size()!=size_t(p.max_points) || prompt.coordinates.size()!=2*prompt.labels.size())
            throw std::invalid_argument("SAM2 prompt count does not match exported model");
        auto coords=std::make_unique<Tensor>(p.T("coords").shape,p.T("coords").type,p.gpu,p.cfg.device);
        auto labels=std::make_unique<Tensor>(p.T("labels").shape,p.T("labels").type,p.gpu,p.cfg.device);
        std::vector<uint8_t> values(coords->storage->bytes);
        for(size_t i=0;i<prompt.coordinates.size();++i) {
            if(!std::isfinite(prompt.coordinates[i])) throw std::invalid_argument("Non-finite SAM2 prompt coordinate");
            PutScalar(values.data(),i,prompt.coordinates[i],p.half);
        }
        p.Upload(coords->Data(),values.data(),values.size());p.Upload(labels->Data(),prompt.labels.data(),labels->storage->bytes);
        prepared->values.emplace("coords",std::move(coords));prepared->values.emplace("labels",std::move(labels));
    } else {
        if(prompt.text.find_first_not_of(" \t\r\n")==std::string::npos) throw std::invalid_argument("SAM3 text prompt is empty");
        const auto tokens=p.tokenizer->Encode(prompt.text);
        if(tokens.size()!=32) throw std::runtime_error("SAM3 text context must be 32");
        p.Upload(p.T("tokens").Data(),tokens.data(),tokens.size()*sizeof(int64_t));p.G("text").Run();
        for(auto& [name,tensor]:p.G("text").outputs) {
            auto value=std::make_unique<Tensor>(tensor->shape,tensor->type,p.gpu,p.cfg.device);
            p.Copy(value->Data(),tensor->Data(),value->storage->bytes);
            prepared->values.emplace(name,std::move(value));
        }
        if(p.gpu) Check(cudaStreamSynchronize(p.stream));
    }
    // Retain prompt allocations until Sam destruction: releasing/replacing a
    // prompt during submission cannot trigger cudaFree or invalidate queued copies.
    p.prompts.push_back(prepared);
    PreparedPrompt result;result.impl_=std::move(prepared);return result;
}
FrameSubmission Sam::RunFrame(const FrameView& source,const MaskOutputView& output,State& destination,
                             const State* previous,const Initialization* initialization) {
    auto& p=*impl_;DeviceScope scope(p.gpu,p.cfg.device);
    FrameView frame=source;
    if(!frame.row_bytes) frame.row_bytes=ptrdiff_t(frame.width)*(frame.format==FrameFormat::Rgb8?3:16);
    auto& state=*destination.impl_;
    p.Validate(frame,output,state,previous);
    if(!previous && (!initialization || !initialization->prompt.impl_ || initialization->prompt.impl_->owner!=&p || !initialization->sequence_length))
        throw std::invalid_argument("Initialization requires a prepared prompt and positive sequence length");
    CompletionToken token;
    for(auto& event:p.events) if(!event->leased) {event->leased=true;token.slot_=event.get();break;}
    if(!token.slot_) throw std::runtime_error("SAM completion capacity exhausted; release old tokens or increase constructor capacity");
    try {
        if(!previous) {
            state.ready=false;state.step=0;state.length=initialization->sequence_length;state.tracking=initialization->tracking;
            state.active=state.next=0;state.width=frame.width;state.height=frame.height;
            state.prompt=initialization->prompt;state.confirmed=state.pending=false;
            if(p.gpu) detail::ResetMeta(state.meta->As<TemporalMeta>(),state.length,p.stream);
            else {*state.meta->As<TemporalMeta>()=TemporalMeta{};state.meta->As<TemporalMeta>()->length=state.length;}
        } else if(previous!=&destination) {
            const auto& src=*previous->impl_;
            p.Copy(state.ring->data,src.ring->data,p.state_bytes);p.Copy(state.meta->data,src.meta->data,sizeof(TemporalMeta));
            state.step=src.step;state.length=src.length;state.active=src.active;state.next=src.next;
            state.width=src.width;state.height=src.height;state.tracking=src.tracking;state.prompt=src.prompt;
            state.confirmed=src.confirmed;state.pending=false;state.ready=false;
        }
        if(p.current_prompt!=state.prompt.impl_.get()) {
            for(auto& [name,tensor]:state.prompt.impl_->values) p.Copy(p.T(name).Data(),tensor->Data(),tensor->storage->bytes);
            p.current_prompt=state.prompt.impl_.get();
        }
        if(p.gpu && p.objects==16) Check(cudaMemsetAsync(p.T("error").Data(),0,sizeof(int),p.stream));
        p.Encode(frame);
        if(p.objects==1) p.Run2(state,previous==nullptr);
        else p.Run3(state);
        p.Dump(state);
        if(state.tracking && p.objects==16) {
            if(p.gpu) Check(cudaMemsetAsync(p.T("boxes").Data(),0,p.T("boxes").storage->bytes,p.stream));
            else std::memset(p.T("boxes").Data(),0,p.T("boxes").storage->bytes);
        }
        if(p.gpu && output.memory==MemoryLocation::Device) {
            detail::WriteMasks(p.T("masks").As(),p.T("valid").As(),p.T("scores").As(),p.T("presence").As(),p.T("boxes").As(),p.objects,p.size,output,p.stream);
        } else {
            if(p.gpu) {
                Check(cudaMemcpyAsync(p.host_masks->data,p.T("masks").Data(),p.T("masks").storage->bytes,cudaMemcpyDeviceToHost,p.stream));
                Check(cudaMemcpyAsync(p.host_valid->data,p.T("valid").Data(),p.objects*4,cudaMemcpyDeviceToHost,p.stream));
                Check(cudaMemcpyAsync(p.host_scores->data,p.T("scores").Data(),p.objects*4,cudaMemcpyDeviceToHost,p.stream));
                Check(cudaMemcpyAsync(p.host_logits->data,p.T("presence").Data(),p.objects*4,cudaMemcpyDeviceToHost,p.stream));
                Check(cudaMemcpyAsync(p.host_boxes->data,p.T("boxes").Data(),p.objects*16,cudaMemcpyDeviceToHost,p.stream));
                Check(cudaStreamSynchronize(p.stream)); // Explicit host-output adapter boundary.
            }
            HostOutput(p.gpu?p.host_masks->As():p.T("masks").As(),p.gpu?p.host_valid->As():p.T("valid").As(),
                       p.gpu?p.host_scores->As():p.T("scores").As(),p.gpu?p.host_logits->As():p.T("presence").As(),
                       p.gpu?p.host_boxes->As():p.T("boxes").As(),p.objects,p.size,output);
        }
        ++state.step;state.ready=true;
        if(p.gpu) {
            if(p.objects==16) Check(cudaMemcpyAsync(token.slot_->diagnostic->data,p.T("error").Data(),sizeof(int),cudaMemcpyDeviceToHost,p.stream));
            else *token.slot_->diagnostic->As<int>()=0;
            Check(cudaEventRecord(token.slot_->event,p.stream));
        }
        return {&destination,std::move(token)};
    } catch(...) {state.ready=false;throw;}
}
} // namespace din::sam
