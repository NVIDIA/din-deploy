// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvtx_helper.h"
#include <nlohmann/json.hpp>

#ifdef DIN_WHISPER_CUDA
#include <cuda_runtime_api.h>

#include "whisper_kernels.h"
#include <onnxruntime_run_options_config_keys.h>
#endif

namespace din::asr::whisper
{
namespace fs = std::filesystem;
namespace
{

// --- IEEE-754 half -> float (reading fp16 logits) -----------------------

float HalfToFloat(uint16_t h)
{
    const uint32_t sign = (h & 0x8000U) << 16;
    const uint32_t exp = (h >> 10) & 0x1FU;
    const uint32_t mant = h & 0x3FFU;
    uint32_t bits = 0;
    if (exp == 0)
    {
        if (mant != 0)
        {
            int e = -1;
            uint32_t m = mant;
            do
            {
                ++e;
                m <<= 1;
            } while ((m & 0x400U) == 0);
            bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFU) << 13);
        }
        else
        {
            bits = sign;
        }
    }
    else if (exp == 0x1FU)
    {
        bits = sign | 0x7F800000U | (mant << 13);
    }
    else
    {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// --- ONNX session IO introspection --------------------------------------

struct IoInfo
{
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

IoInfo InputInfo(Ort::Session& session, const std::string& name)
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        if (name == session.GetInputNameAllocated(i, allocator).get())
        {
            const auto info = session.GetInputTypeInfo(i);
            const auto shape_info = info.GetTensorTypeAndShapeInfo();
            return {shape_info.GetShape(), shape_info.GetElementType()};
        }
    }
    throw std::runtime_error("model input not found: " + name);
}

IoInfo OutputInfo(Ort::Session& session, const std::string& name)
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        if (name == session.GetOutputNameAllocated(i, allocator).get())
        {
            const auto info = session.GetOutputTypeInfo(i);
            const auto shape_info = info.GetTensorTypeAndShapeInfo();
            return {shape_info.GetShape(), shape_info.GetElementType()};
        }
    }
    throw std::runtime_error("model output not found: " + name);
}

int CountOutputsWithPrefix(Ort::Session& session, const std::string& prefix)
{
    Ort::AllocatorWithDefaultOptions allocator;
    int count = 0;
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        const std::string name = session.GetOutputNameAllocated(i, allocator).get();
        if (name.rfind(prefix, 0) == 0)
        {
            ++count;
        }
    }
    return count;
}

// --- dtype-aware tensor helpers -----------------------------------------

// Allocates one self-KV cache tensor [1, heads, max_positions, head_dim] in the
// ONNX element type for the model's IO precision.
ONNXTensorElementDataType IoElementType(bool io_fp16)
{
    return io_fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// True if the EP device is an NVIDIA GPU (PCI vendor 0x10DE), i.e. eligible for
// the CUDA argmax kernel.
constexpr uint32_t kNvidiaVendorId = 0x10DE;
bool IsNvidiaGpu(const Ort::ConstEpDevice& ep_device)
{
    if (!ep_device)
    {
        return false;
    }
    const auto hardware = ep_device.Device();
    return hardware.Type() == OrtHardwareDeviceType_GPU && hardware.VendorId() == kNvidiaVendorId;
}

// Allocates a tensor of the model's IO precision, on device or host.
Ort::Value MakeIoValue(OrtRunner& runner, const std::vector<int64_t>& shape, bool io_fp16, bool device)
{
    const auto type = IoElementType(io_fp16);
    if (device)
    {
        auto allocator = runner.DeviceAllocator();
        return Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
    }
    static Ort::AllocatorWithDefaultOptions cpu;
    return Ort::Value::CreateTensor(cpu, shape.data(), shape.size(), type);
}

Ort::Value MakeSelfKv(OrtRunner& runner, const ModelDims& dims, bool device)
{
    return MakeIoValue(runner, {1, dims.num_heads, dims.max_positions, dims.head_dim}, dims.io_fp16, device);
}

// Allocates a tensor of the model's IO precision in pinned host memory (device
// provider) or default host memory (CPU), for fast host<->device staging.
Ort::Value MakePinnedValue(OrtRunner& runner, const std::vector<int64_t>& shape, bool io_fp16, bool device)
{
    const auto type = IoElementType(io_fp16);
    if (device)
    {
        auto allocator = runner.PinnedAllocator();
        return Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
    }
    static Ort::AllocatorWithDefaultOptions cpu;
    return Ort::Value::CreateTensor(cpu, shape.data(), shape.size(), type);
}

size_t SelfKvBytes(const ModelDims& dims)
{
    const size_t elems = static_cast<size_t>(dims.num_heads) * dims.max_positions * dims.head_dim;
    return elems * (dims.io_fp16 ? sizeof(uint16_t) : sizeof(float));
}

// Argmax over the final sequence position of logits, restricted to [lower, upper).
int32_t ArgmaxLastPosition(const Ort::Value& logits, int64_t lower, int64_t upper)
{
    const auto info = logits.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const int64_t seq = shape[1];
    const int64_t vocab = shape[2];
    const size_t offset = static_cast<size_t>((seq - 1) * vocab);
    int32_t best = lower;
    float best_val = -1e30F;
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        const auto* row = reinterpret_cast<const uint16_t*>(logits.GetTensorData<Ort::Float16_t>()) + offset;
        for (int64_t id = lower; id < upper; ++id)
        {
            const float v = HalfToFloat(row[id]);
            if (v > best_val)
            {
                best_val = v;
                best = id;
            }
        }
    }
    else
    {
        const float* row = logits.GetTensorData<float>() + offset;
        for (int64_t id = lower; id < upper; ++id)
        {
            if (row[id] > best_val)
            {
                best_val = row[id];
                best = id;
            }
        }
    }
    return best;
}

}  // namespace

WhisperPipeline::WhisperPipeline(WhisperConfig config)
    : config_(std::move(config))
    , env_(ORT_LOGGING_LEVEL_WARNING, "din_asr_whisper")
{
    DIN_NVTX_FUNC_RANGE();
    const fs::path model_dir = config_.model_dir;
    const auto vocab_path = model_dir / "vocab.json";
    if (!std::filesystem::exists(vocab_path))
    {
        throw std::runtime_error("missing vocab.json at: " + vocab_path.string());
    }
    tokenizer_ = std::make_unique<din::io::Tokenizer>(vocab_path.string(), din::io::TokenizerFormat::WhisperJson);

    if (config_.provider == "trt-rtx" || config_.provider == "trt")
    {
        din::common::RegisterTensorRTRTXProvider(env_);
        compute_stream_ = din::common::CreateTensorRTRTXComputeStream(env_);
    }

    EpContextOptions ep_context;
    ep_context.output_dir = config_.ep_context_dir.string();

    // All exported shapes are static, so no TRT optimization profile is needed;
    // only the cache subpath and the (disabled) CUDA-graph flag matter. The
    // self-KV cache is re-bound each token, which CUDA-graph capture can't replay.
    // The cache subpath is model-specific so different sizes don't collide on a
    // shared EP-context engine.
    const std::string tag = config_.model_dir.filename().string();
    ModelProfile mel_profile;
    mel_profile.cache_subpath = tag + "_mel";
    mel_profile.enable_cuda_graph = false;
    ModelProfile encoder_profile;
    encoder_profile.cache_subpath = tag + "_encoder";
    encoder_profile.enable_cuda_graph = false;
    encoder_profile.embed_ep_context = false;  // large-v3 fp32 engines exceed the 2 GB embed limit
    ModelProfile decoder_profile;
    decoder_profile.cache_subpath = tag + "_decoder";
    decoder_profile.enable_cuda_graph = false;
    decoder_profile.embed_ep_context = false;

    mel_ = MakeRunner(model_dir / "mel.onnx", ep_context, mel_profile);
    encoder_ = MakeRunner(model_dir / "encoder.onnx", ep_context, encoder_profile);
    decoder_ = MakeRunner(model_dir / "decoder.onnx", ep_context, decoder_profile);
    use_device_io_ = mel_->HasDeviceIo() && encoder_->HasDeviceIo() && decoder_->HasDeviceIo();

    DetectModel();

    // Control-token ids: defaults are the 99-language layout; read exact ids from
    // added_tokens.json when present (large-v3 shifts the task tokens by +1).
    const fs::path added = model_dir / "added_tokens.json";
    if (fs::exists(added))
    {
        std::ifstream stream(added);
        const nlohmann::json tokens = nlohmann::json::parse(stream);
        const auto get = [&](const char* key, int64_t fallback)
        {
            return tokens.contains(key) ? tokens.at(key).get<int64_t>() : fallback;
        };
        special_.sot = get("<|startoftranscript|>", special_.sot);
        special_.transcribe = get("<|transcribe|>", special_.transcribe);
        special_.notimestamps = get("<|notimestamps|>", special_.notimestamps);
        special_.eot = get("<|endoftext|>", special_.eot);
        special_.lang_first = special_.sot + 1;
        special_.lang_last = special_.transcribe - 2;  // ..langs.., <|translate|>, <|transcribe|>
    }

    // Stable IO name storage, referenced by IoBinding throughout decoding.
    for (int i = 0; i < dims_.num_layers; ++i)
    {
        const auto s = std::to_string(i);
        enc_cross_key_out_.push_back("present_key_cross_" + s);
        enc_cross_value_out_.push_back("present_value_cross_" + s);
        dec_past_self_key_in_.push_back("past_key_self_" + s);
        dec_past_self_value_in_.push_back("past_value_self_" + s);
        dec_past_cross_key_in_.push_back("past_key_cross_" + s);
        dec_past_cross_value_in_.push_back("past_value_cross_" + s);
        dec_present_self_key_out_.push_back("present_key_self_" + s);
        dec_present_self_value_out_.push_back("present_value_self_" + s);
    }

    if (std::getenv("DIN_WHISPER_DEBUG"))
    {
        std::cerr << "[debug] dims: layers=" << dims_.num_layers << " heads=" << dims_.num_heads
                  << " head_dim=" << dims_.head_dim << " frames=" << dims_.encoder_frames
                  << " max_pos=" << dims_.max_positions << " vocab=" << dims_.vocab
                  << " io=" << (dims_.io_fp16 ? "fp16" : "fp32") << " | tokens: sot=" << special_.sot
                  << " transcribe=" << special_.transcribe << " notimestamps=" << special_.notimestamps << " lang=["
                  << special_.lang_first << "," << special_.lang_last << "]" << std::endl;
    }

    SetupEncodePath();
    if (use_device_io_)
    {
        SetupDecodePath();
    }
}

void WhisperPipeline::DetectModel()
{
    dims_.io_fp16 = InputInfo(encoder_->session, "audio_features").type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    const auto cross = OutputInfo(encoder_->session, "present_key_cross_0").shape;  // [1, heads, frames, head_dim]
    dims_.num_heads = static_cast<int>(cross.at(1));
    dims_.encoder_frames = static_cast<int>(cross.at(2));
    dims_.head_dim = static_cast<int>(cross.at(3));
    dims_.num_layers = CountOutputsWithPrefix(encoder_->session, "present_key_cross_");
    dims_.max_positions = static_cast<int>(InputInfo(decoder_->session, "past_key_self_0").shape.at(2));
    dims_.vocab = OutputInfo(decoder_->session, "logits").shape.at(2);
    if (dims_.num_layers <= 0 || dims_.num_heads <= 0)
    {
        throw std::runtime_error("failed to detect Whisper model geometry");
    }
}

std::unique_ptr<OrtRunner> WhisperPipeline::MakeRunner(const fs::path& path, const EpContextOptions& ep_context,
                                                       const ModelProfile& profile)
{
    if (!std::filesystem::exists(path))
    {
        throw std::runtime_error(path.string() + " does not exist");
    }
    auto* stream = (config_.provider == "trt-rtx" || config_.provider == "trt") ? &compute_stream_ : nullptr;
    return std::make_unique<OrtRunner>(env_, path.string(), config_.provider, config_.ep_cache_dir.string(), ep_context,
                                       profile, stream);
}

void WhisperPipeline::SetupEncodePath()
{
    // All mel/encoder shapes are fixed, so allocate the device buffers and their
    // IoBindings once and reuse them for every chunk. samples are staged through
    // pinned host memory (TensorBuffer) and copied to the device with CopyTensor.
    const auto af_shape = InputInfo(encoder_->session, "audio_features").shape;        // [1, n_mels, 3000]
    const auto hs_shape = OutputInfo(encoder_->session, "hidden_states").shape;        // [1, 1500, hidden]
    const auto ck_shape = OutputInfo(encoder_->session, "present_key_cross_0").shape;  // [1, heads, 1500, head_dim]

    samples_.emplace(*mel_, std::vector<int64_t>{1, kChunkSamples}, use_device_io_);
    audio_features_ = MakeIoValue(*mel_, af_shape, dims_.io_fp16, use_device_io_);
    hidden_states_ = MakeIoValue(*encoder_, hs_shape, dims_.io_fp16, use_device_io_);
    cross_kv_.clear();
    cross_kv_.reserve(2 * dims_.num_layers);
    for (int i = 0; i < 2 * dims_.num_layers; ++i)
    {
        cross_kv_.push_back(MakeIoValue(*encoder_, ck_shape, dims_.io_fp16, use_device_io_));
    }

    mel_binding_.emplace(mel_->session);
    mel_binding_->BindInput("samples", samples_->BindingValue());
    mel_binding_->BindOutput("audio_features", audio_features_);

    encoder_binding_.emplace(encoder_->session);
    encoder_binding_->BindInput("audio_features", audio_features_);
    encoder_binding_->BindOutput("hidden_states", hidden_states_);
    for (int i = 0; i < dims_.num_layers; ++i)
    {
        encoder_binding_->BindOutput(enc_cross_key_out_[i].c_str(), cross_kv_[2 * i]);
        encoder_binding_->BindOutput(enc_cross_value_out_[i].c_str(), cross_kv_[2 * i + 1]);
    }
}

void WhisperPipeline::EncodeChunk(const std::vector<float>& samples)
{
    din::common::nvtx_scoped_range range{"encode"};
    // Stage the 30 s window into pinned host memory and copy it to the device
    // sample buffer (no-op copy on the CPU provider).
    std::copy_n(samples.data(), static_cast<size_t>(kChunkSamples), samples_->HostData());
    samples_->CopyAsyncToDevice();
    Ort::RunOptions run_options;
    run_options.SetSyncStream(compute_stream_);
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    mel_->session.Run(run_options, *mel_binding_);          // samples -> audio_features_ (device)
    encoder_->session.Run(run_options, *encoder_binding_);  // audio_features_ -> cross_kv_ (in place)
}

void WhisperPipeline::SetupDecodePath()
{
    // Persistent decode IO (device path). Self-KV cache aliases past==present in
    // place; the three scalar inputs are pinned; logits land in pinned host memory.
    self_kv_.clear();
    self_kv_.reserve(2 * dims_.num_layers);
    for (int i = 0; i < 2 * dims_.num_layers; ++i)
    {
        self_kv_.push_back(MakeSelfKv(*decoder_, dims_, /*device=*/true));
    }
    // We disable UMA for these since we make use of the double buffering on a CPU and GPU tensor
    // to submit async to the GPU while already overwriting the host value for the next inference.
    dec_input_ids_.emplace(*decoder_, std::vector<int64_t>{1, 1}, /*use_device_io=*/true, /*disable_uma=*/ true);
    dec_write_idx_.emplace(*decoder_, std::vector<int64_t>{1}, /*use_device_io=*/true, /*disable_uma=*/ true);
    dec_nonpad_.emplace(*decoder_, std::vector<int64_t>{1}, /*use_device_io=*/true, /*disable_uma=*/ true);

    // Runtime choice: use the CUDA argmax kernel only when it was compiled in, the
    // decoder's EP device is an NVIDIA GPU, and it wasn't disabled on the CLI;
    // otherwise argmax on the CPU. device_is_cuda_ (without the CLI gate) also
    // decides whether the self-KV cache is cleared with cudaMemset.
    device_is_cuda_ = false;
#ifdef DIN_WHISPER_CUDA
    device_is_cuda_ = IsNvidiaGpu(decoder_->ep_device);
#endif
    use_cuda_sampling_ = device_is_cuda_ && !config_.disable_cuda_sampling;

    if (use_cuda_sampling_)
    {
        // Logits stay on-device for the kernel; only the winning token is copied
        // back through the pinned token buffer.
        dec_logits_ = MakeIoValue(*decoder_, {1, 1, dims_.vocab}, dims_.io_fp16, /*device=*/true);
    }
    else
    {
        // CPU argmax needs the logits on the host; land them in pinned memory.
        dec_logits_ = MakePinnedValue(*decoder_, {1, 1, dims_.vocab}, dims_.io_fp16, /*device=*/true);
    }

    decoder_binding_.emplace(decoder_->session);
    decoder_binding_->BindInput("input_ids", dec_input_ids_->BindingValue());
    decoder_binding_->BindInput("write_indices", dec_write_idx_->BindingValue());
    decoder_binding_->BindInput("nonpad_kv_seqlen", dec_nonpad_->BindingValue());
    for (int i = 0; i < dims_.num_layers; ++i)
    {
        decoder_binding_->BindInput(dec_past_self_key_in_[i].c_str(), self_kv_[2 * i]);
        decoder_binding_->BindInput(dec_past_self_value_in_[i].c_str(), self_kv_[2 * i + 1]);
        decoder_binding_->BindInput(dec_past_cross_key_in_[i].c_str(), cross_kv_[2 * i]);
        decoder_binding_->BindInput(dec_past_cross_value_in_[i].c_str(), cross_kv_[2 * i + 1]);
        decoder_binding_->BindOutput(dec_present_self_key_out_[i].c_str(), self_kv_[2 * i]);
        decoder_binding_->BindOutput(dec_present_self_value_out_[i].c_str(), self_kv_[2 * i + 1]);
    }
    decoder_binding_->BindOutput("logits", dec_logits_);

    if (std::getenv("DIN_WHISPER_DEBUG"))
    {
        std::cerr << "[debug] greedy argmax: " << (use_cuda_sampling_ ? "cuda kernel" : "cpu (pinned logits)")
                  << std::endl;
    }
}

void WhisperPipeline::ZeroSelfKv()
{
    // Uninitialized FLOAT16 slots can be NaN, which would poison the masked
    // attention; clear the persistent cache once per chunk on the compute stream.
    const size_t bytes = SelfKvBytes(dims_);
#ifdef DIN_WHISPER_CUDA
    if (device_is_cuda_)
    {
        // CUDA device memory: zero it in place instead of copying a host buffer.
        auto stream = reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
        for (auto& kv : self_kv_)
        {
            void* ptr = dims_.io_fp16 ? static_cast<void*>(kv.GetTensorMutableData<Ort::Float16_t>())
                                      : static_cast<void*>(kv.GetTensorMutableData<float>());
            const cudaError_t status = cudaMemsetAsync(ptr, 0, bytes, stream);
            if (status != cudaSuccess)
            {
                throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(status));
            }
        }
    }
    else
#endif
    {
        // Fallback (no CUDA, or non-CUDA device memory): copy a host zero buffer in.
        const std::vector<int64_t> shape{1, dims_.num_heads, dims_.max_positions, dims_.head_dim};
        std::vector<uint8_t> host_zeros(bytes, 0);
        auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value zero_src = Ort::Value::CreateTensor(cpu, host_zeros.data(), host_zeros.size(), shape.data(),
                                                       shape.size(), IoElementType(dims_.io_fp16));
        for (auto& kv : self_kv_)
        {
            decoder_->env.CopyTensor(zero_src, kv, compute_stream_);
        }
    }
}

void WhisperPipeline::ArgmaxLogits(int64_t lower, int64_t upper)
{
#ifdef DIN_WHISPER_CUDA
    if (use_cuda_sampling_)
    {
        // Argmax on the GPU over the on-device logits; only the token crosses back.
        const void* logits_dev = dims_.io_fp16 ? static_cast<const void*>(dec_logits_.GetTensorData<Ort::Float16_t>())
                                               : static_cast<const void*>(dec_logits_.GetTensorData<float>());
        int32_t* out = dec_input_ids_->BindingValue().GetTensorMutableData<int32_t>();
        auto stream = reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
        launch_whisper_argmax(stream, logits_dev, dims_.io_fp16, lower, upper, out);
        token_ready_notification_ = std::move(dec_input_ids_->CopyAsyncToHostWithNotification());
    }
    else
#endif
    {
        // CPU fallback: logits were copied to pinned host memory by the decoder
        // Run. Unlike the CUDA kernel path, writing the staging buffer does not
        // update the decoder's device input, so upload the selected token before
        // the next decode step consumes it.
        dec_input_ids_->HostData()[0] = ArgmaxLastPosition(dec_logits_, lower, upper);
        token_ready_notification_ = std::move(dec_input_ids_->CopyAsyncToDeviceWithNotification());
        // Decode runs with its default stream in this mode; wait here rather than
        // relying on a later notification wait, which occurs after the next Run.
        token_ready_notification_.Sync();
    }
}

std::vector<int64_t> WhisperPipeline::DecodeChunkDevice(int64_t& lang_token)
{
    ZeroSelfKv();
    int64_t pos = 0;
    const auto step = [&](int32_t token, bool update_token = true)
    {
        din::common::nvtx_scoped_range range{"decode_step"};
        if (update_token)
        {
            dec_input_ids_->HostData()[0] = token;
            dec_input_ids_->CopyAsyncToDevice();
        }
        dec_write_idx_->HostData()[0] = pos;
        dec_write_idx_->CopyAsyncToDevice();
        dec_nonpad_->HostData()[0] = pos + 1;
        auto upload_event = dec_nonpad_->CopyAsyncToDeviceWithNotification();
        // ^ we only need a single event here since all are enqueued on the same stream
        Ort::RunOptions run_options;
#ifdef DIN_WHISPER_CUDA
        if (use_cuda_sampling_)
        {
            // The argmax kernel + token copy synchronize the step, so let Run
            // enqueue asynchronously instead of blocking here. When argmaxing on
            // the CPU we keep the default sync so the pinned logits are ready.
            run_options.SetSyncStream(compute_stream_);
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        }
#endif
        decoder_->session.Run(run_options, *decoder_binding_);
        upload_event.Sync();  // ensure that the host buffer can be rewritten on the next iteration
        ++pos;
    };
    step(static_cast<int32_t>(special_.sot));
    if (config_.lang_id == "auto")
    {
        ArgmaxLogits(special_.lang_first, special_.lang_last + 1);
        token_ready_notification_.Sync();
        lang_token = dec_input_ids_->HostData()[0];
    }
    else
    {
        lang_token = ResolveLanguageToken(dec_logits_);
    }
    step(static_cast<int32_t>(lang_token), !use_cuda_sampling_);
    ArgmaxLogits(0, special_.eot + 1);
    token_ready_notification_.Sync();
    step(static_cast<int32_t>(special_.transcribe));
    step(static_cast<int32_t>(special_.notimestamps));

    std::vector<int64_t> generated;
    // for GPU we call step once more than required since synchronous execution is slower
    // than just submitting one more inference than required
    do
    {
        ArgmaxLogits(0, special_.eot + 1);
        step(dec_input_ids_->HostData()[0], !use_cuda_sampling_);
        token_ready_notification_.Sync();
        generated.push_back(dec_input_ids_->HostData()[0]);
    } while (dec_input_ids_->HostData()[0] != special_.eot && pos < dims_.max_positions);
    return generated;
}

std::vector<int64_t> WhisperPipeline::DecodeChunk(int64_t& lang_token)
{
    if (use_device_io_)
    {
        return DecodeChunkDevice(lang_token);
    }
    return DecodeChunkHost(lang_token);
}

std::vector<int64_t> WhisperPipeline::DecodeChunkHost(int64_t& lang_token)
{
    std::vector<Ort::Value>& cross_kv = cross_kv_;
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Fixed-capacity self-KV cache, zero-initialized (stale slots beyond the write
    // position are masked by nonpad_kv_seqlen, but zeroing avoids NaN poisoning).
    std::vector<Ort::Value> self_kv;
    self_kv.reserve(2 * dims_.num_layers);
    for (int i = 0; i < 2 * dims_.num_layers; ++i)
    {
        self_kv.push_back(MakeSelfKv(*decoder_, dims_, use_device_io_));
    }
    const size_t kv_bytes = SelfKvBytes(dims_);
    {
        // Host path (CPU provider): zero the freshly-allocated cache in place.
        for (auto& kv : self_kv)
        {
            void* raw = dims_.io_fp16 ? static_cast<void*>(kv.GetTensorMutableData<Ort::Float16_t>())
                                      : static_cast<void*>(kv.GetTensorMutableData<float>());
            std::memset(raw, 0, kv_bytes);
        }
    }

    // Device (TRT) requires past/present self-KV to alias (in-place update); the
    // CPU path feeds present->past out-of-place, matching the reference.
    const bool alias = use_device_io_;
    int64_t pos = 0;
    const auto step = [&](int32_t token) -> Ort::Value
    {
        din::common::nvtx_scoped_range range{"decode_step"};
        const std::vector<int64_t> id_shape{1, 1};
        Ort::Value ids = Ort::Value::CreateTensor<int32_t>(cpu, &token, 1, id_shape.data(), id_shape.size());
        const std::vector<int64_t> scalar_shape{1};
        int64_t write_index = pos;
        int64_t nonpad_len = pos + 1;
        Ort::Value write_indices =
            Ort::Value::CreateTensor<int64_t>(cpu, &write_index, 1, scalar_shape.data(), scalar_shape.size());
        Ort::Value nonpad =
            Ort::Value::CreateTensor<int64_t>(cpu, &nonpad_len, 1, scalar_shape.data(), scalar_shape.size());

        Ort::IoBinding binding(decoder_->session);
        binding.BindInput("input_ids", ids);
        binding.BindInput("write_indices", write_indices);
        binding.BindInput("nonpad_kv_seqlen", nonpad);
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            binding.BindInput(dec_past_self_key_in_[i].c_str(), self_kv[2 * i]);
            binding.BindInput(dec_past_self_value_in_[i].c_str(), self_kv[2 * i + 1]);
            binding.BindInput(dec_past_cross_key_in_[i].c_str(), cross_kv[2 * i]);
            binding.BindInput(dec_past_cross_value_in_[i].c_str(), cross_kv[2 * i + 1]);
        }
        binding.BindOutput("logits", cpu);  // logits always read on host for argmax
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            if (alias)
            {
                binding.BindOutput(dec_present_self_key_out_[i].c_str(), self_kv[2 * i]);
                binding.BindOutput(dec_present_self_value_out_[i].c_str(), self_kv[2 * i + 1]);
            }
            else
            {
                binding.BindOutput(dec_present_self_key_out_[i].c_str(), cpu);
                binding.BindOutput(dec_present_self_value_out_[i].c_str(), cpu);
            }
        }
        Ort::RunOptions run_options;
        decoder_->session.Run(run_options, binding);

        auto out_names = binding.GetOutputNames();
        auto out_values = binding.GetOutputValues();
        std::unordered_map<std::string, size_t> index;
        for (size_t i = 0; i < out_names.size(); ++i)
        {
            index.emplace(out_names[i], i);
        }
        if (!alias)
        {
            for (int i = 0; i < dims_.num_layers; ++i)
            {
                self_kv[2 * i] = std::move(out_values[index.at(dec_present_self_key_out_[i])]);
                self_kv[2 * i + 1] = std::move(out_values[index.at(dec_present_self_value_out_[i])]);
            }
        }
        ++pos;
        return std::move(out_values[index.at("logits")]);
    };

    // Prefill the forced prompt token by token; the language tag is chosen from
    // the logits that follow <|startoftranscript|>.
    Ort::Value logits = step(static_cast<int32_t>(special_.sot));
    lang_token = ResolveLanguageToken(logits);
    step(static_cast<int32_t>(lang_token));
    step(static_cast<int32_t>(special_.transcribe));
    logits = step(static_cast<int32_t>(special_.notimestamps));

    const bool debug = std::getenv("DIN_WHISPER_DEBUG") != nullptr;
    std::vector<int64_t> generated;
    int64_t next = ArgmaxLastPosition(logits, 0, special_.eot + 1);
    while (next != special_.eot && pos < dims_.max_positions)
    {
        generated.push_back(next);
        if (debug && generated.size() <= 8)
        {
            std::cerr << "[debug] pos=" << pos << " token=" << next << std::endl;
        }
        logits = step(static_cast<int32_t>(next));
        next = ArgmaxLastPosition(logits, 0, special_.eot + 1);
    }
    return generated;
}

int64_t WhisperPipeline::ResolveLanguageToken(const Ort::Value& sot_logits) const
{
    if (config_.lang_id == "auto")
    {
        return ArgmaxLastPosition(sot_logits, special_.lang_first, special_.lang_last + 1);
    }
    for (int64_t id = special_.lang_first; id <= special_.lang_last; ++id)
    {
        if (tokenizer_->LanguageCode(id) == config_.lang_id)
        {
            return id;
        }
    }
    return special_.lang_first;  // default to the first language on an unknown code
}

TranscriptionResult WhisperPipeline::Transcribe(const Audio& audio)
{
    DIN_NVTX_FUNC_RANGE();
    if (audio.sample_rate != kSampleRate)
    {
        throw std::runtime_error("expected 16 kHz audio");
    }

    TranscriptionResult result;
    result.audio_seconds = audio.Duration();
    encode_seconds_ = 0.0;
    greedy_seconds_ = 0.0;

    const auto transcribe_start = std::chrono::steady_clock::now();

    const size_t total = audio.samples.size();
    const size_t num_chunks = (total + kChunkSamples - 1) / kChunkSamples;

    // Optional cap (testing): DIN_WHISPER_MAX_CHUNKS limits processed windows.
    size_t max_chunks = num_chunks;
    if (const char* cap = std::getenv("DIN_WHISPER_MAX_CHUNKS"))
    {
        max_chunks = std::min<size_t>(num_chunks, std::strtoul(cap, nullptr, 10));
    }

    std::string text;
    size_t chunk_index = 0;
    for (size_t start = 0; start < total && chunk_index < max_chunks; start += kChunkSamples, ++chunk_index)
    {
        std::vector<float> chunk(kChunkSamples, 0.0F);
        const size_t count = std::min<size_t>(kChunkSamples, total - start);
        std::copy_n(audio.samples.begin() + start, count, chunk.begin());

        const auto encode_start = std::chrono::steady_clock::now();
        EncodeChunk(chunk);  // mel + encoder -> persistent cross_kv_
        encode_seconds_ += SecondsSince(encode_start);

        const auto greedy_start = std::chrono::steady_clock::now();
        int64_t lang_token = special_.lang_first;
        const std::vector<int64_t> tokens = DecodeChunk(lang_token);
        if (result.language.empty())
        {
            result.language = tokenizer_->LanguageCode(lang_token);
        }
        greedy_seconds_ += SecondsSince(greedy_start);

        std::string chunk_text = tokenizer_->Decode(tokens);
        const auto first = chunk_text.find_first_not_of(" \t\r\n");
        const auto last = chunk_text.find_last_not_of(" \t\r\n");
        chunk_text = first == std::string::npos ? std::string{} : chunk_text.substr(first, last - first + 1);
        if (!text.empty() && !chunk_text.empty())
        {
            text += ' ';
        }
        text += chunk_text;

        std::cerr << "[chunk " << (chunk_index + 1) << "/" << num_chunks << " " << result.language << "] " << chunk_text
                  << std::endl;
    }

    result.text = text;
    result.encode_seconds = encode_seconds_;
    result.greedy_seconds = greedy_seconds_;
    result.transcribe_seconds = SecondsSince(transcribe_start);
    return result;
}

TranscriptionResult WhisperPipeline::TranscribeFile(const fs::path& audio_path)
{
    const auto audio = din::io::LoadAudio(audio_path.string(), kSampleRate);
    return Transcribe(audio);
}

void WhisperPipeline::Print(std::ostream& stream, const TranscriptionResult& result,
                            const TranscriptionOptions& /*options*/) const
{
    if (!result.language.empty())
    {
        stream << "[language: " << result.language << "]\n";
    }
    stream << result.text << '\n';
}

}  // namespace din::asr::whisper
