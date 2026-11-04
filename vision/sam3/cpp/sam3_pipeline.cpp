// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include "sam3_tokenizer.h"
#include "sam3_cuda.h"
#include "sam3_artifact.h"
#include "sam3_nvtx.h"
#include "ort_session.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <deque>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

namespace din::sam3 {
namespace {
Ort::Env& Environment() {
    return din::common::SharedInferenceEnvironment();
}
using Values = std::map<std::string, Ort::Value>;
using Inputs = std::map<std::string, const Ort::Value*>;
using Clock = std::chrono::steady_clock;
thread_local std::map<std::string, double>* active_timings = nullptr;
struct TimingScope {
    const char* name;

    Clock::time_point start = Clock::now();
    bool stopped = false;
    void Stop() { if (!stopped && active_timings) (*active_timings)[name] += std::chrono::duration<double, std::milli>(Clock::now() - start).count(); stopped = true; }
    ~TimingScope() { Stop(); }
};
struct FrameTiming {
    std::map<std::string, double>* previous = active_timings;
    FrameTiming(std::map<std::string, double>& timings, bool enabled) {
        timings.clear(); active_timings = enabled ? &timings : nullptr;
    }
    ~FrameTiming() {active_timings = previous;}
};
void Cuda(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
// Views do not own storage: the graph/pool owns it until pipeline destruction.
Ort::Value View(const Ort::Value& value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    return Ort::Value::CreateTensor(value.GetTensorMemoryInfo(), const_cast<void*>(value.GetTensorRawData()),
        value.GetTensorSizeInBytes(),
        shape.data(), shape.size(), info.GetElementType());
}
class Graph {
public:
    static std::filesystem::path ModelPath(const Sam3PipelineConfig& config, const std::string& name) {
        if (name == "tracking_decoder" && !config.diagnostic_decoder.empty()) return config.diagnostic_decoder;
        if (!config.diagnostic_graph_dir.empty()) {
            const auto override = config.diagnostic_graph_dir / (name + ".onnx");
            if (std::filesystem::exists(override)) return override;
        }
        return config.model_dir / (name + ".onnx");
    }
    Graph(const Sam3PipelineConfig& config, const std::string& name, Ort::SyncStream* stream)
        : name_("graph." + name), runner_(Environment(), ModelPath(config, name).string(), config.provider,
                  ArtifactCache(config.cache_dir, ModelPath(config, name)).string(),
                  {(ArtifactCache(config.cache_dir, ModelPath(config, name)) / "context").string()}, Profile(name), stream) {
        Ort::AllocatorWithDefaultOptions allocator;
        if (name == "memory_attention" && !config.diagnostic_dir.empty()) {
            std::filesystem::create_directories(config.diagnostic_dir);
            binding_log_.open(config.diagnostic_dir / "memory_attention_bindings.jsonl", std::ios::trunc);
            binding_log_.exceptions(std::ios::badbit | std::ios::failbit);
        }
        for (size_t i = 0; i < runner_.session.GetInputCount(); ++i) {
            const std::string input = runner_.session.GetInputNameAllocated(i, allocator).get();
            // The pinned detector export declares these unused positions but has
            // no consumers for them. ORT treats unused inputs as CPU inputs and
            // otherwise downloads 106 MB/frame while binding them.
            if (name == "concept_detector" && (input == "detector_position_0" || input == "detector_position_1")) continue;
            inputs_.push_back(input);
        }
        for (size_t i = 0; i < runner_.session.GetOutputCount(); ++i) outputs_.emplace_back(runner_.session.GetOutputNameAllocated(i, allocator).get());
        binding_ = Ort::IoBinding(runner_.session);
        if (runner_.HasDeviceIo()) {
            std::ifstream metadata_file(config.model_dir / "metadata.json");
            const auto metadata = nlohmann::json::parse(metadata_file);
            for (size_t i = 0; i < outputs_.size(); ++i) {
                auto type_info = runner_.session.GetOutputTypeInfo(i);
                auto info = type_info.GetTensorTypeAndShapeInfo();
                auto shape = info.GetShape();
                if (std::any_of(shape.begin(), shape.end(), [](auto d) { return d <= 0; })) {
                    for (const auto& output : metadata.at("graphs").at(name).at("outputs"))
                        if (output.at("name") == outputs_[i]) shape = output.at("shape").get<std::vector<int64_t>>();
                }
                if (std::any_of(shape.begin(), shape.end(), [](auto d) { return d <= 0; }))
                    throw std::runtime_error("SAM3 requires static graph outputs: " + outputs_[i]);
                auto value = Ort::Value::CreateTensor(runner_.DeviceAllocator(), shape.data(), shape.size(), info.GetElementType());
                binding_.BindOutput(outputs_[i].c_str(), value);
                output_storage_.emplace(outputs_[i], std::move(value));
            }
        }
        if (config.profile && stream) {
            compute_stream_ = static_cast<cudaStream_t>(stream->GetHandle());
            Cuda(cudaEventCreate(&gpu_start_));
            const auto status = cudaEventCreate(&gpu_end_);
            if (status != cudaSuccess) {cudaEventDestroy(gpu_start_); Cuda(status);}
        }
    }
    ~Graph() {if (gpu_start_) cudaEventDestroy(gpu_start_); if (gpu_end_) cudaEventDestroy(gpu_end_);}
    static din::common::ModelProfile Profile(const std::string& name) {
        din::common::ModelProfile profile;
        profile.cache_subpath = name;
        profile.embed_ep_context = false;
        profile.enable_cuda_graph = false;
        profile.extra_ep_options.emplace_back("nv_persistent_context_memory", "1");
        if (name == "image_preprocess") {
            profile.min_shapes = "image_rgb:1x1x1x3";
            profile.opt_shapes = "image_rgb:1x1008x1008x3";
            profile.max_shapes = "image_rgb:1x8192x8192x3";
        }
        if (name == "memory_attention") {
            for (const auto* input : {"memory_image", "memory_image_position", "memory", "memory_position", "object_pointers", "pointer_position"}) {
                const bool pointer = std::string(input).find("pointer") != std::string::npos;
                const int tokens = pointer ? 16 : 5184;
                auto append = [&](std::string& shapes, int count) {
                    if (!shapes.empty()) shapes += ',';
                    shapes += std::string(input) + ':' + std::to_string(tokens * count) + "x1x256";
                };
                append(profile.min_shapes, 1); append(profile.opt_shapes, 2); append(profile.max_shapes, pointer ? 16 : 10);
            }
        }
        return profile;
    }
    Values Run(const Inputs& inputs) {
        if (binding_log_.is_open()) {
            nlohmann::json record{{"call", call_index_++}};
            for (const auto& name : inputs_) {
                const auto& value = *inputs.at(name);
                record["inputs"][name] = {{"shape", value.GetTensorTypeAndShapeInfo().GetShape()},
                    {"address", reinterpret_cast<uintptr_t>(value.GetTensorRawData())}};
            }
            for (const auto& [name, value] : output_storage_)
                record["outputs"][name] = {{"shape", value.GetTensorTypeAndShapeInfo().GetShape()},
                    {"address", reinterpret_cast<uintptr_t>(value.GetTensorRawData())}};
            binding_log_ << record.dump() << '\n';
        }
        TraceRange trace{name_.c_str()};
        TimingScope timing{name_.c_str()};
        const auto bind_name = "bind." + name_;
        TimingScope bind_time{bind_name.c_str()};
        auto& binding = binding_;
        binding.ClearBoundInputs();
        for (const auto& name : inputs_) binding.BindInput(name.c_str(), *inputs.at(name));
        const auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        for (const auto& name : outputs_) {
            if (!runner_.HasDeviceIo()) binding.BindOutput(name.c_str(), cpu);
        }
        bind_time.Stop();
        if (gpu_start_) Cuda(cudaEventRecord(gpu_start_, compute_stream_));
        const auto run_start = Clock::now();

        Ort::RunOptions options;
        if (runner_.HasDeviceIo()) options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        runner_.session.Run(options, binding);

        if (active_timings) (*active_timings)["run." + name_] += std::chrono::duration<double, std::milli>(Clock::now() - run_start).count();
        if (gpu_end_) {Cuda(cudaEventRecord(gpu_end_, compute_stream_)); timing_pending_ = true;}
        if (!runner_.HasDeviceIo()) binding.SynchronizeOutputs();
        Values result;
        if (runner_.HasDeviceIo()) {
            for (const auto& [name, value] : output_storage_) result.emplace(name, View(value));
        } else {
            auto values = binding.GetOutputValues();
            for (size_t i = 0; i < values.size(); ++i) result.emplace(outputs_[i], std::move(values[i]));
        }
        return result;
    }
    void CollectTimings() {
        if (timing_pending_ && active_timings) {
            Cuda(cudaEventQuery(gpu_end_)); // The frame's readback has already waited.
            float milliseconds = 0;
            Cuda(cudaEventElapsedTime(&milliseconds, gpu_start_, gpu_end_));
            (*active_timings)["gpu_stream." + name_] += milliseconds;
            timing_pending_ = false;
        }
    }
    din::common::OrtRunner& Runner() {return runner_;}
private:
    std::string name_;
    std::ofstream binding_log_;
    size_t call_index_ = 0;
    cudaStream_t compute_stream_ = nullptr;
    cudaEvent_t gpu_start_ = nullptr, gpu_end_ = nullptr;
    bool timing_pending_ = false;
    din::common::OrtRunner runner_;
    Values output_storage_;
    Ort::IoBinding binding_{nullptr};
    std::vector<std::string> inputs_, outputs_;
};
class PinnedReadback {
    float* data_ = nullptr;
    size_t capacity_ = 0;
    cudaEvent_t ready_ = nullptr;
public:
    ~PinnedReadback() { if (ready_) cudaEventDestroy(ready_); if (data_) cudaFreeHost(data_); }
    std::vector<float> Read(const Ort::Value& value, cudaStream_t stream) {

        auto info = value.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("Expected FP32 SAM3 graph output");
        std::vector<float> result(info.GetElementCount());
        if (value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU)
            std::copy_n(value.GetTensorData<float>(), result.size(), result.data());
        else {
            if (result.size() > capacity_) {
                float* replacement = nullptr;
                Cuda(cudaMallocHost(reinterpret_cast<void**>(&replacement), result.size() * sizeof(float)));
                if (data_) cudaFreeHost(data_);
                data_ = replacement; capacity_ = result.size();
            }
            {
              Cuda(cudaMemcpyAsync(data_, value.GetTensorData<float>(), result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream)); }
            if (!ready_) Cuda(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
            Cuda(cudaEventRecord(ready_, stream));
            {  Cuda(cudaEventSynchronize(ready_)); }

            std::copy_n(data_, result.size(), result.data());
        }
        return result;
    }
};
float Sigmoid(float x) {return x >= 0 ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x));}
}

struct Sam3Pipeline::Impl {
    Sam3PipelineConfig config;
    TextTokenizer tokenizer;
    Ort::SyncStream stream{nullptr};
    PinnedReadback readback;
    struct Buffer {
        std::unique_ptr<din::common::TensorBuffer<float>> storage;
        Ort::Value view{nullptr};
        size_t capacity = 0;
        float* HostData() {return storage->HostData();}
        Ort::Value& BindingValue() {return view;}
        void CopyAsyncToDevice() {storage->CopyAsyncToDevice();}
    };
    std::map<std::string, Buffer> buffers;
    std::unique_ptr<din::common::TensorBuffer<uint8_t>> rgb_storage;
    uint32_t rgb_width = 0, rgb_height = 0;
    Values history_storage;
    std::array<const float*, 16> final_masks{};
    RasterizedMasks rasters;
    struct RasterStorage {
        void* device = nullptr;
        void* host = nullptr;
        size_t capacity = 0;
        cudaEvent_t ready = nullptr;
        ~RasterStorage() {if (ready) cudaEventDestroy(ready); if (device) cudaFree(device); if (host) cudaFreeHost(host);}
    } raster_storage;
    std::unique_ptr<Graph> preprocess, vision, text, detector;
    std::string cached_text;
    std::vector<int64_t> token_ids;
    Values text_features;
    Values sequence_visual;
    Values sequence_detection;
    std::vector<size_t> sequence_queries;
    bool sequence_ready = false;
    TextPrompt sequence_prompt;
    size_t sequence_frames = 0, sequence_index = 0;
    uint32_t sequence_width = 0, sequence_height = 0;
    std::unique_ptr<Graph> seed, seed_memory, tracking, tracking_memory, attention;
    struct MemoryFrame { size_t index; Values memory; Ort::Value image{nullptr}, position{nullptr}, pointers{nullptr}; };
    std::deque<MemoryFrame> memories;
    std::vector<Instance> tracked;
    nlohmann::json temporal;
    std::map<std::string, double> timings;
    ~Impl() {if (stream) cudaStreamSynchronize(static_cast<cudaStream_t>(stream.GetHandle()));}
    void CollectTimings() {
        for (auto* graph : {preprocess.get(), vision.get(), text.get(), detector.get(), seed.get(), seed_memory.get(), tracking.get(), tracking_memory.get(), attention.get()})
            if (graph) graph->CollectTimings();
    }
    FrameResult Finish(FrameResult result) {
        if (config.rasterize_output && vision->Runner().HasDeviceIo()) {

            const size_t pixels = size_t(result.width) * result.height;
            auto& storage = raster_storage;
            if (storage.capacity < pixels * 21) {
                if (storage.device) {Cuda(cudaFree(storage.device)); storage.device = nullptr;}
                if (storage.host) {Cuda(cudaFreeHost(storage.host)); storage.host = nullptr;}
                Cuda(cudaMalloc(&storage.device, pixels * 21));
                Cuda(cudaMallocHost(&storage.host, pixels * 21));
                storage.capacity = pixels * 21;
            }
            if (!storage.ready) Cuda(cudaEventCreateWithFlags(&storage.ready, cudaEventDisableTiming));
            MaskRasterInputs inputs;
            rasters.ids.clear();
            for (const auto& instance : result.instances) {
                if (instance.id < 1 || instance.id > 16 || !final_masks[instance.id - 1]) throw std::logic_error("Missing device mask");
                inputs.probabilities[inputs.count] = final_masks[instance.id - 1];
                inputs.ids[inputs.count++] = instance.id;
                rasters.ids.push_back(instance.id);
            }
            auto* labels = static_cast<uint32_t*>(storage.device);
            auto* combined = reinterpret_cast<uint8_t*>(labels + pixels);
            auto* individual = combined + pixels;
            RasterizeMasks(inputs, pixels, labels, combined, individual, stream.GetHandle());
            const auto cuda_stream = static_cast<cudaStream_t>(stream.GetHandle());
            {
              Cuda(cudaMemcpyAsync(storage.host, storage.device, pixels * (5 + inputs.count), cudaMemcpyDeviceToHost, cuda_stream)); }
            Cuda(cudaEventRecord(storage.ready, cuda_stream));
            {  Cuda(cudaEventSynchronize(storage.ready)); }
            auto* host_labels = static_cast<uint32_t*>(storage.host);
            auto* host_combined = reinterpret_cast<uint8_t*>(host_labels + pixels);
            rasters.width = result.width; rasters.height = result.height;
            rasters.labels.assign(host_labels, host_labels + pixels);
            rasters.combined.assign(host_combined, host_combined + pixels);
            rasters.individual.assign(host_combined + pixels, host_combined + pixels * (1 + inputs.count));
        }
        CollectTimings();
        return result;
    }
    std::vector<float> ReadFloat(const Ort::Value& value) {
        return readback.Read(value, stream ? static_cast<cudaStream_t>(stream.GetHandle()) : nullptr);
    }
    std::vector<std::vector<float>> ReadMetadata(std::initializer_list<const Ort::Value*> values) {

        std::vector<std::vector<float>> result;
        if (!vision->Runner().HasDeviceIo()) {
            for (auto* value : values) result.push_back(ReadFloat(*value));
            return result;
        }
        size_t count = 0;
        for (auto* value : values) count += value->GetTensorTypeAndShapeInfo().GetElementCount();
        auto& packed = Scratch("metadata_readback", {int64_t(count)});
        size_t offset = 0;
        for (auto* value : values) {
            const auto n = value->GetTensorTypeAndShapeInfo().GetElementCount();
            Cuda(cudaMemcpyAsync(packed.BindingValue().GetTensorMutableData<float>() + offset, value->GetTensorData<float>(),
                n * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.GetHandle())));
            offset += n;
        }
        const auto host = ReadFloat(packed.BindingValue());
        offset = 0;
        for (auto* value : values) {
            const auto n = value->GetTensorTypeAndShapeInfo().GetElementCount();
            result.emplace_back(host.begin() + offset, host.begin() + offset + n); offset += n;
        }
        return result;
    }
    Buffer& Scratch(const std::string& name, std::vector<int64_t> shape, size_t reserve = 0) {
        size_t count = 1; for (auto d : shape) count *= size_t(d);
        auto& buffer = buffers[name];
        auto& runner = vision->Runner();
        if (!buffer.storage || buffer.capacity < count) {

            buffer.capacity = std::max(count, reserve);
            buffer.storage = std::make_unique<din::common::TensorBuffer<float>>(runner,
                std::vector<int64_t>{int64_t(buffer.capacity)}, runner.HasDeviceIo());
            // Spare capacity is copied by TensorBuffer uploads; initialize it too.
            std::fill_n(buffer.storage->HostData(), buffer.capacity, 0.f);
        }
        auto& backing = buffer.storage->BindingValue();
        buffer.view = Ort::Value::CreateTensor<float>(backing.GetTensorMemoryInfo(),
            backing.GetTensorMutableData<float>(), count, shape.data(), shape.size());
        return buffer;
    }
    Ort::Value SaveHistory(const std::string& key, const Ort::Value& source) {

        auto& stored = history_storage.try_emplace(key, nullptr).first->second;
        const auto info = source.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (!stored) stored = Ort::Value::CreateTensor(vision->Runner().DeviceAllocator(), shape.data(), shape.size(), info.GetElementType());
        Cuda(cudaMemcpyAsync(stored.GetTensorMutableData<float>(), source.GetTensorData<float>(),
            info.GetElementCount() * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.GetHandle())));
        return View(stored);
    }
    explicit Impl(Sam3PipelineConfig cfg) : config(std::move(cfg)), tokenizer(config.model_dir / "tokenizer.json") {

        if (config.provider != "cpu" && config.provider != "trt-rtx") throw std::invalid_argument("Provider must be cpu or trt-rtx");
        if (!std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0 || config.confidence_threshold > 1)
            throw std::invalid_argument("Confidence threshold must be in [0, 1]");
        std::ifstream file(config.model_dir / "metadata.json");
        auto metadata = nlohmann::json::parse(file);
        if (metadata.at("schema_version") != 1 || metadata.at("model_family") != "sam3.1" || metadata.at("dtype") != "float32")
            throw std::runtime_error("Unsupported SAM3 artifact contract");
        if (metadata.at("image_size") != 1008 || metadata.at("backbone_stride") != 14 || metadata.at("max_objects") != 16 ||
            metadata.at("multiplex_count") != 16 || metadata.at("text_context_length") != 32 || metadata.at("hotstart_delay") != 0)
            throw std::runtime_error("SAM3 artifacts have incompatible spatial, capacity, text or causal settings");
        for (const auto* graph : {"image_preprocess", "vision_encoder", "text_encoder", "concept_detector"})
            if (!metadata.at("graphs").contains(graph)) throw std::runtime_error(std::string("Missing SAM3 graph: ") + graph);
        if (config.provider == "trt-rtx") {
            din::common::RegisterTensorRTRTXProvider(Environment());
            stream = din::common::CreateTensorRTRTXComputeStream(Environment());
        }
        auto* compute = config.provider == "trt-rtx" ? &stream : nullptr;
        if (config.provider == "cpu") preprocess = std::make_unique<Graph>(config, "image_preprocess", compute);
        vision = std::make_unique<Graph>(config, "vision_encoder", compute);
        text = std::make_unique<Graph>(config, "text_encoder", compute);
        detector = std::make_unique<Graph>(config, "concept_detector", compute);
    }
    FrameResult Segment(const Image& image, const TextPrompt& prompt, const DeviceFrame* device = nullptr, bool tracking_detection = false) {
        TimingScope segment_time{"segment_total"};
        if (image.width == 0 || image.height == 0 || image.width > 8192 || image.height > 8192 || (!device && image.rgb.size() != size_t(image.width) * image.height * 3))
            throw std::invalid_argument("Invalid SAM3 RGB image (maximum dimension 8192)");
        if (device && (config.provider != "trt-rtx" || !device->rgb || device->row_bytes < size_t(image.width) * 3 || device->row_bytes > PTRDIFF_MAX / image.height))
            throw std::invalid_argument("Device frames require trt-rtx and valid RGB8 device storage");
        if (prompt.text.find_first_not_of(" \t\r\n") == std::string::npos) throw std::invalid_argument("Text prompt must not be empty");
        auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        if (text_features.empty() || cached_text != prompt.text) {

            token_ids = tokenizer.Encode(prompt.text);
            const std::array<int64_t, 2> shape{1, static_cast<int64_t>(token_ids.size())};
            auto tokens = Ort::Value::CreateTensor<int64_t>(cpu, token_ids.data(), token_ids.size(), shape.data(), shape.size());
            text_features = text->Run({{"token_ids", &tokens}});
            cached_text = prompt.text;
        }
        // Upload once; subsequent graph connections bind the existing device tensors.
        auto& runner = vision->Runner();
        if (!device) {

            if (!rgb_storage || rgb_width != image.width || rgb_height != image.height) {
                rgb_storage = std::make_unique<din::common::TensorBuffer<uint8_t>>(runner, std::vector<int64_t>{1, image.height, image.width, 3}, runner.HasDeviceIo());
                rgb_width = image.width; rgb_height = image.height;
            }
            std::copy(image.rgb.begin(), image.rgb.end(), rgb_storage->HostData());
            rgb_storage->CopyAsyncToDevice();
        } else {

            cudaPointerAttributes attributes{};
            Cuda(cudaPointerGetAttributes(&attributes, device->rgb));
            int current_device = 0;
            Cuda(cudaGetDevice(&current_device));
            if (attributes.type != cudaMemoryTypeDevice || attributes.device != current_device)
                throw std::invalid_argument("Device frame must reside on the pipeline CUDA device");
            cudaEvent_t ready = nullptr;
            Cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            const auto recorded = cudaEventRecord(ready, static_cast<cudaStream_t>(device->producer_stream));
            const auto waited = recorded == cudaSuccess ? cudaStreamWaitEvent(static_cast<cudaStream_t>(stream.GetHandle()), ready, 0) : recorded;
            cudaEventDestroy(ready);
            Cuda(waited);
        }
        Values visual;
        if (runner.HasDeviceIo()) {
            auto& scratch = Scratch("preprocess_scratch", {image.height, 1008, 3});
            auto& prepared = Scratch("prepared", {1, 3, 1008, 1008});

            PreprocessDeviceRgb(device ? device->rgb : rgb_storage->BindingValue().GetTensorData<uint8_t>(), image.width, image.height, device ? device->row_bytes : image.width * 3,
                                scratch.BindingValue().GetTensorMutableData<float>(), prepared.BindingValue().GetTensorMutableData<float>(), stream.GetHandle());

            visual = vision->Run({{"image", &prepared.BindingValue()}});
        } else {
            auto prepared = preprocess->Run({{"image_rgb", &rgb_storage->BindingValue()}});
            visual = vision->Run({{"image", &prepared.at("image")}});
        }
        Inputs detector_inputs;
        for (const auto& [name, value] : text_features) detector_inputs.emplace(name, &value);
        for (const auto& [name, value] : visual) detector_inputs.emplace(name, &value);
        auto detection = detector->Run(detector_inputs);
        TimingScope postprocess_time{"detection_postprocess"};
        const auto masks = runner.HasDeviceIo() ? std::vector<float>{} : ReadFloat(detection.at("mask_logits"));
        const auto metadata = ReadMetadata({&detection.at("boxes_xyxy"), &detection.at("class_logits"), &detection.at("presence_logits")});
        const auto& boxes = metadata[0];
        const auto& logits = metadata[1];
        const auto& presence = metadata[2];
        const auto shape = detection.at("mask_logits").GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[0] != 1 || size_t(shape[1]) != logits.size() || presence.size() != 1 || boxes.size() != logits.size() * 4)
            throw std::runtime_error("Inconsistent SAM3 detector outputs");
        const auto finite = [](float value) { return std::isfinite(value); };
        if (!std::isfinite(presence[0]) || !std::all_of(logits.begin(), logits.end(), finite) || !std::all_of(boxes.begin(), boxes.end(), finite))
            throw std::runtime_error("SAM3 detector produced non-finite scores or boxes");
        const int mh = static_cast<int>(shape[2]), mw = static_cast<int>(shape[3]);
        const auto selected_count = std::count_if(logits.begin(), logits.end(), [&](float logit) {
            return Sigmoid(logit) * Sigmoid(presence[0]) > config.confidence_threshold;
        });
        if (selected_count > 16) throw std::runtime_error("SAM3 object capacity exceeded: " + std::to_string(selected_count) + " detections, capacity 16");
        FrameResult result{image.width, image.height, {}};
        if (sequence_ready) sequence_queries.clear();
        for (size_t query = 0; query < logits.size(); ++query) {
            const float score = Sigmoid(logits[query]) * Sigmoid(presence[0]);
            if (score <= config.confidence_threshold) continue;
            if (sequence_ready) sequence_queries.push_back(query);
            Instance instance;
            instance.id = static_cast<uint32_t>(result.instances.size() + 1); instance.score = score;
            for (size_t k = 0; k < 4; ++k) instance.box[k] = boxes[query * 4 + k] * (k % 2 ? image.height : image.width);
            // Tracking consumes detector logits directly on device. Export only
            // the final tracked masks, not discarded full-resolution detections.
            if (tracking_detection && runner.HasDeviceIo()) {result.instances.push_back(std::move(instance)); continue;}
            instance.probabilities.resize(size_t(image.width) * image.height);
            if (runner.HasDeviceIo()) {
                auto& probabilities = Scratch("output_mask_" + std::to_string(instance.id), {image.height, image.width});
                final_masks[instance.id - 1] = probabilities.BindingValue().GetTensorData<float>();
                ResizeMaskProbability(detection.at("mask_logits").GetTensorData<float>() + query * mh * mw, mw, mh,
                                      probabilities.BindingValue().GetTensorMutableData<float>(), image.width, image.height, stream.GetHandle());
                instance.probabilities = ReadFloat(probabilities.BindingValue());
                if (!std::all_of(instance.probabilities.begin(), instance.probabilities.end(), finite))
                    throw std::runtime_error("SAM3 decoder produced non-finite masks");
                result.instances.push_back(std::move(instance));
                continue;
            }
            const float* mask = masks.data() + query * mh * mw;
            for (uint32_t y = 0; y < image.height; ++y) {
                const float fy = std::max(0.f, (y + .5f) * mh / image.height - .5f);
                const int y0 = std::min(int(fy), mh - 1), y1 = std::min(y0 + 1, mh - 1);
                for (uint32_t x = 0; x < image.width; ++x) {
                    const float fx = std::max(0.f, (x + .5f) * mw / image.width - .5f);
                    const int x0 = std::min(int(fx), mw - 1), x1 = std::min(x0 + 1, mw - 1);
                    const float a = std::lerp(mask[y0 * mw + x0], mask[y0 * mw + x1], fx - x0);
                    const float b = std::lerp(mask[y1 * mw + x0], mask[y1 * mw + x1], fx - x0);
                    instance.probabilities[size_t(y) * image.width + x] = Sigmoid(std::lerp(a, b, fy - y0));
                }
            }
            if (!std::all_of(instance.probabilities.begin(), instance.probabilities.end(), finite))
                throw std::runtime_error("SAM3 decoder produced non-finite masks");
            result.instances.push_back(std::move(instance));
        }
        if (sequence_ready) { sequence_visual = std::move(visual); sequence_detection = std::move(detection); }
        return result;
    }
    void Reset(const TextPrompt& prompt, size_t count) {

        if (!count || prompt.text.empty()) throw std::invalid_argument("Sequence requires a concept and frame count");
        std::ifstream constants(config.model_dir / "temporal_constants.json");
        temporal = nlohmann::json::parse(constants);
        if (!seed) {
            auto* compute = config.provider == "trt-rtx" ? &stream : nullptr;
            seed = std::make_unique<Graph>(config, "seed_decoder", compute);
            seed_memory = std::make_unique<Graph>(config, "seed_memory", compute);
            tracking = std::make_unique<Graph>(config, "tracking_decoder", compute);
            tracking_memory = std::make_unique<Graph>(config, "tracking_memory", compute);
            attention = std::make_unique<Graph>(config, "memory_attention", compute);
        }
        memories.clear(); tracked.clear(); sequence_visual.clear(); sequence_detection.clear(); sequence_queries.clear();
        sequence_prompt = prompt; sequence_frames = count; sequence_index = 0;
        sequence_width = sequence_height = 0; sequence_ready = true;
    }
    FrameResult Next(const Image& image, const DeviceFrame* device = nullptr);
};
#include "sam3_sequence.inc"
Sam3Pipeline::Sam3Pipeline(Sam3PipelineConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Sam3Pipeline::~Sam3Pipeline() = default;
FrameResult Sam3Pipeline::SegmentImage(const Image& image, const TextPrompt& prompt) {
    FrameTiming timing(impl_->timings, impl_->config.profile); return impl_->Finish(impl_->Segment(image, prompt));
}
FrameResult Sam3Pipeline::SegmentDeviceFrame(const DeviceFrame& frame, const TextPrompt& prompt) {
    FrameTiming timing(impl_->timings, impl_->config.profile);
    return impl_->Finish(impl_->Segment(Image{frame.width, frame.height, {}}, prompt, &frame));
}
void Sam3Pipeline::ResetSequence(const TextPrompt& prompt, size_t frame_count) { impl_->Reset(prompt, frame_count); }
FrameResult Sam3Pipeline::ProcessNextFrame(const Image& image) {
    FrameTiming timing(impl_->timings, impl_->config.profile); return impl_->Finish(impl_->Next(image));
}
FrameResult Sam3Pipeline::ProcessDeviceFrame(const DeviceFrame& frame) {
    FrameTiming timing(impl_->timings, impl_->config.profile);
    return impl_->Finish(impl_->Next(Image{frame.width, frame.height, {}}, &frame));
}
const std::map<std::string, double>& Sam3Pipeline::LastTimings() const {return impl_->timings;}
const RasterizedMasks* Sam3Pipeline::LastRasterizedMasks() const {
    return impl_->config.rasterize_output && impl_->vision->Runner().HasDeviceIo() ? &impl_->rasters : nullptr;
}
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
