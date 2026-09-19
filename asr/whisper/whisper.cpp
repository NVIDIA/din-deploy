// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

float LastPositionLogit(const Ort::Value& logits, int64_t id)
{
    const auto info = logits.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const size_t offset = static_cast<size_t>((shape.at(1) - 1) * shape.at(2) + id);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        return HalfToFloat(reinterpret_cast<const uint16_t*>(logits.GetTensorData<Ort::Float16_t>())[offset]);
    }
    return logits.GetTensorData<float>()[offset];
}

float LastPositionProbability(const Ort::Value& logits, int64_t token, int64_t vocab)
{
    float maximum = -INFINITY;
    for (int64_t id = 0; id < vocab; ++id)
    {
        maximum = std::max(maximum, LastPositionLogit(logits, id));
    }
    double sum = 0.0;
    for (int64_t id = 0; id < vocab; ++id)
    {
        sum += std::exp(static_cast<double>(LastPositionLogit(logits, id) - maximum));
    }
    return static_cast<float>(std::exp(static_cast<double>(LastPositionLogit(logits, token) - maximum)) / sum);
}

int32_t SelectTimestampToken(const Ort::Value& logits, const std::vector<int64_t>& generated,
                             const SpecialTokens& special, float* selected_logprob = nullptr)
{
    const int64_t vocab = logits.GetTensorTypeAndShapeInfo().GetShape().at(2);
    int64_t minimum_timestamp = special.timestamp_first;
    bool allow_text = true;
    bool allow_timestamps = true;

    if (generated.empty())
    {
        allow_text = false;
    }
    else if (generated.back() >= special.timestamp_first)
    {
        const bool penultimate_was_timestamp = generated.size() < 2 ||
            generated[generated.size() - 2] >= special.timestamp_first;
        minimum_timestamp = generated.back() + (penultimate_was_timestamp ? 1 : 0);
        allow_timestamps = !penultimate_was_timestamp;
        allow_text = penultimate_was_timestamp;
    }
    else
    {
        for (auto it = generated.rbegin(); it != generated.rend(); ++it)
        {
            if (*it >= special.timestamp_first)
            {
                minimum_timestamp = *it;
                break;
            }
        }
    }

    const int64_t maximum_timestamp = generated.empty() ? std::min(vocab, special.timestamp_first + 51) : vocab;
    float best_text = -INFINITY;
    int32_t best_text_id = static_cast<int32_t>(special.eot);
    double text_exp_sum = 0.0;
    const auto accumulate = [](float value, float& maximum, double& exp_sum)
    {
        if (value > maximum)
        {
            exp_sum = std::isfinite(maximum) ? exp_sum * std::exp(static_cast<double>(maximum - value)) + 1.0 : 1.0;
            maximum = value;
        }
        else if (std::isfinite(value))
        {
            exp_sum += std::exp(static_cast<double>(value - maximum));
        }
    };
    if (allow_text)
    {
        for (int64_t id = 0; id <= special.eot && id < vocab; ++id)
        {
            const float value = LastPositionLogit(logits, id);
            if (value > best_text)
            {
                best_text_id = static_cast<int32_t>(id);
            }
            accumulate(value, best_text, text_exp_sum);
        }
    }
    else if (!generated.empty())
    {
        best_text = LastPositionLogit(logits, special.eot);
        text_exp_sum = 1.0;
    }

    float best_timestamp = -INFINITY;
    int32_t best_timestamp_id = static_cast<int32_t>(special.timestamp_first);
    double timestamp_exp_sum = 0.0;
    if (allow_timestamps)
    {
        for (int64_t id = std::max(minimum_timestamp, special.timestamp_first); id < maximum_timestamp; ++id)
        {
            const float value = LastPositionLogit(logits, id);
            if (value > best_timestamp)
            {
                best_timestamp_id = static_cast<int32_t>(id);
            }
            accumulate(value, best_timestamp, timestamp_exp_sum);
        }
    }
    const float timestamp_logprob = timestamp_exp_sum > 0.0
                                        ? best_timestamp + static_cast<float>(std::log(timestamp_exp_sum))
                                        : -INFINITY;
    const bool force_timestamp = timestamp_logprob > best_text || !allow_text;
    const bool selected_timestamp = force_timestamp || best_timestamp > best_text;
    if (selected_logprob != nullptr)
    {
        const float timestamp_lse = timestamp_exp_sum > 0.0
                                        ? best_timestamp + static_cast<float>(std::log(timestamp_exp_sum))
                                        : -INFINITY;
        const float text_lse = text_exp_sum > 0.0 ? best_text + static_cast<float>(std::log(text_exp_sum)) : -INFINITY;
        float denominator = timestamp_lse;
        if (!force_timestamp)
        {
            if (!std::isfinite(timestamp_lse))
            {
                denominator = text_lse;
            }
            else if (std::isfinite(text_lse))
            {
                const float maximum = std::max(text_lse, timestamp_lse);
                denominator = maximum + std::log(std::exp(text_lse - maximum) + std::exp(timestamp_lse - maximum));
            }
        }
        const float selected_value = selected_timestamp ? best_timestamp : best_text;
        *selected_logprob = selected_value - denominator;
    }
    return selected_timestamp ? best_timestamp_id : best_text_id;
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

    // One decoder_prefill.onnx session handles both multi-token prefill and S=1
    // autoregressive decode. Keep separate TensorRT profiles so neither phase
    // compromises the other's optimized shape.
    const std::string tag = config_.model_dir.filename().string();
    ModelProfile mel_profile;
    mel_profile.cache_subpath = tag + "_mel";
    mel_profile.enable_cuda_graph = false;
    ModelProfile encoder_profile;
    encoder_profile.cache_subpath = tag + "_encoder";
    encoder_profile.enable_cuda_graph = false;
    encoder_profile.embed_ep_context = false;  // large-v3 fp32 engines exceed the 2 GB embed limit
    const std::string decode_shapes = "input_ids:1x1,write_indices:1";
    const std::string prefill_shapes = "input_ids:1x" + std::to_string(kMaxPrefillTokens) +
        ",write_indices:" + std::to_string(kMaxPrefillTokens);
    ModelProfile decoder_profile;
    decoder_profile.min_shapes = decode_shapes + "," + decode_shapes;
    decoder_profile.opt_shapes = decode_shapes + "," + prefill_shapes;
    decoder_profile.max_shapes = decode_shapes + "," + prefill_shapes;
    decoder_profile.cache_subpath = tag + "_decoder_prefill_profiles_1_" + std::to_string(kMaxPrefillTokens);
    decoder_profile.enable_cuda_graph = false;
    decoder_profile.embed_ep_context = false;
    decoder_profile.extra_ep_options.emplace_back("nv_multi_profile_enable", "1");

    mel_ = MakeRunner(model_dir / "mel.onnx", ep_context, mel_profile);
    encoder_ = MakeRunner(model_dir / "encoder.onnx", ep_context, encoder_profile);
    decoder_ = MakeRunner(model_dir / "decoder_prefill.onnx", ep_context, decoder_profile);
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
        special_.start_of_prev = get("<|startofprev|>", special_.start_of_prev);
        special_.no_speech = get("<|nospeech|>", special_.no_speech);
        special_.timestamp_first = get("<|0.00|>", special_.timestamp_first);
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
    dec_input_ids_.emplace(*decoder_, std::vector<int64_t>{1, 1}, /*use_device_io=*/true, /*disable_uma=*/true);
    dec_write_idx_.emplace(*decoder_, std::vector<int64_t>{1}, /*use_device_io=*/true, /*disable_uma=*/true);
    dec_nonpad_.emplace(*decoder_, std::vector<int64_t>{1}, /*use_device_io=*/true, /*disable_uma=*/true);

    // Runtime choice: use the CUDA argmax kernel only when it was compiled in, the
    // decoder's EP device is an NVIDIA GPU, and it wasn't disabled on the CLI;
    // otherwise argmax on the CPU. device_is_cuda_ (without the CLI gate) also
    // decides whether the self-KV cache is cleared with cudaMemset.
    device_is_cuda_ = false;
#ifdef DIN_WHISPER_CUDA
    device_is_cuda_ = IsNvidiaGpu(decoder_->ep_device);
#endif
    // Timestamp rules need the full logits row. Long-form keeps inference on the
    // EP but selects tokens from pinned host logits until the CUDA filter exists.
    use_cuda_sampling_ = device_is_cuda_ && !config_.disable_cuda_sampling && !config_.long_form;

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

void WhisperPipeline::ArgmaxLogits(const Ort::Value& logits, int64_t lower, int64_t upper)
{
#ifdef DIN_WHISPER_CUDA
    if (use_cuda_sampling_)
    {
        // Argmax the final sequence row on the GPU; only the token crosses back.
        const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        const size_t row = static_cast<size_t>(shape.at(1) - 1) * static_cast<size_t>(shape.at(2));
        const void* logits_dev =
            dims_.io_fp16
                ? static_cast<const void*>(logits.GetTensorData<Ort::Float16_t>() + row)
                : static_cast<const void*>(logits.GetTensorData<float>() + row);
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
        dec_input_ids_->HostData()[0] = ArgmaxLastPosition(logits, lower, upper);
        token_ready_notification_ = std::move(dec_input_ids_->CopyAsyncToDeviceWithNotification());
        // Decode runs with its default stream in this mode; wait here rather than
        // relying on a later notification wait, which occurs after the next Run.
        token_ready_notification_.Sync();
    }
}

void WhisperPipeline::SelectTimestampLogits(const Ort::Value& logits, const std::vector<int64_t>& generated)
{
    float selected_logprob = 0.0F;
    const int32_t selected = SelectTimestampToken(logits, generated, special_, &selected_logprob);
    dec_input_ids_->HostData()[0] = selected;
    decoded_sum_logprob_ += selected_logprob;
    ++decoded_token_count_;
    token_ready_notification_ = std::move(dec_input_ids_->CopyAsyncToDeviceWithNotification());
    token_ready_notification_.Sync();
}

std::vector<int64_t> WhisperPipeline::DecodeChunkDevice(int64_t& lang_token, const std::vector<int64_t>& prompt,
                                                        bool timestamps)
{
    decoded_sum_logprob_ = 0.0;
    decoded_token_count_ = 0;
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
        run_options.AddConfigEntry("nv_profile_index", "0");
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

    // Populate several consecutive cache positions in one decoder invocation.
    // The same dynamic decoder_prefill.onnx session is subsequently called with
    // S=1 by step() for autoregressive generation.
    const auto prefill = [&](const std::vector<int32_t>& tokens, int64_t lower, int64_t upper,
                             const std::vector<int64_t>* generated = nullptr, bool capture_no_speech = false)
    {
        if (tokens.empty() || tokens.size() > static_cast<size_t>(kMaxPrefillTokens) ||
            pos + static_cast<int64_t>(tokens.size()) > dims_.max_positions)
        {
            throw std::runtime_error("invalid Whisper prefill token count");
        }

        const int64_t sequence = static_cast<int64_t>(tokens.size());
        din::common::TensorBuffer<int32_t> input_ids(*decoder_, {1, sequence}, true, true);
        din::common::TensorBuffer<int64_t> write_indices(*decoder_, {sequence}, true, true);
        din::common::TensorBuffer<int64_t> nonpad(*decoder_, {1}, true, true);
        std::copy(tokens.begin(), tokens.end(), input_ids.HostData());
        for (int64_t i = 0; i < sequence; ++i)
        {
            write_indices.HostData()[i] = pos + i;
        }
        nonpad.HostData()[0] = pos + sequence;
        input_ids.CopyAsyncToDevice();
        write_indices.CopyAsyncToDevice();
        auto upload_event = nonpad.CopyAsyncToDeviceWithNotification();

        Ort::Value logits = use_cuda_sampling_
                                ? MakeIoValue(*decoder_, {1, sequence, dims_.vocab}, dims_.io_fp16, true)
                                : MakePinnedValue(*decoder_, {1, sequence, dims_.vocab}, dims_.io_fp16, true);
        Ort::IoBinding binding(decoder_->session);
        binding.BindInput("input_ids", input_ids.BindingValue());
        binding.BindInput("write_indices", write_indices.BindingValue());
        binding.BindInput("nonpad_kv_seqlen", nonpad.BindingValue());
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            binding.BindInput(dec_past_self_key_in_[i].c_str(), self_kv_[2 * i]);
            binding.BindInput(dec_past_self_value_in_[i].c_str(), self_kv_[2 * i + 1]);
            binding.BindInput(dec_past_cross_key_in_[i].c_str(), cross_kv_[2 * i]);
            binding.BindInput(dec_past_cross_value_in_[i].c_str(), cross_kv_[2 * i + 1]);
            binding.BindOutput(dec_present_self_key_out_[i].c_str(), self_kv_[2 * i]);
            binding.BindOutput(dec_present_self_value_out_[i].c_str(), self_kv_[2 * i + 1]);
        }
        binding.BindOutput("logits", logits);

        Ort::RunOptions run_options;
        run_options.AddConfigEntry("nv_profile_index", "1");
#ifdef DIN_WHISPER_CUDA
        if (use_cuda_sampling_)
        {
            run_options.SetSyncStream(compute_stream_);
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        }
#endif
        decoder_->session.Run(run_options, binding);
        upload_event.Sync();
        pos += sequence;
        if (capture_no_speech)
        {
            no_speech_probability_ = LastPositionProbability(logits, special_.no_speech, dims_.vocab);
        }
        if (generated != nullptr)
        {
            SelectTimestampLogits(logits, *generated);
        }
        else
        {
            ArgmaxLogits(logits, lower, upper);
        }
        token_ready_notification_.Sync();
        return static_cast<int64_t>(dec_input_ids_->HostData()[0]);
    };

    std::vector<int32_t> prefix;
    if (!prompt.empty())
    {
        prefix.push_back(static_cast<int32_t>(special_.start_of_prev));
        const size_t keep = std::min<size_t>(prompt.size(), kMaxPrefillTokens - 2);
        for (size_t i = prompt.size() - keep; i < prompt.size(); ++i)
        {
            prefix.push_back(static_cast<int32_t>(prompt[i]));
        }
    }
    prefix.push_back(static_cast<int32_t>(special_.sot));
    const int64_t detected = prefill(prefix, special_.lang_first, special_.lang_last + 1, nullptr, true);
    if (lang_token < 0)
    {
        lang_token = config_.lang_id == "auto" ? detected : ResolveLanguageToken(dec_logits_);
    }

    std::vector<int32_t> task_prompt{static_cast<int32_t>(lang_token), static_cast<int32_t>(special_.transcribe)};
    if (!timestamps)
    {
        task_prompt.push_back(static_cast<int32_t>(special_.notimestamps));
    }

    std::vector<int64_t> generated;
    int64_t next = prefill(task_prompt, 0, special_.eot + 1, timestamps ? &generated : nullptr);
    while (next != special_.eot && pos < dims_.max_positions)
    {
        generated.push_back(next);
        step(static_cast<int32_t>(next), !use_cuda_sampling_);
        if (timestamps)
        {
            SelectTimestampLogits(dec_logits_, generated);
        }
        else
        {
            ArgmaxLogits(dec_logits_, 0, special_.eot + 1);
        }
        token_ready_notification_.Sync();
        next = dec_input_ids_->HostData()[0];
    }
    return generated;
}

std::vector<int64_t> WhisperPipeline::DecodeChunk(int64_t& lang_token, const std::vector<int64_t>& prompt,
                                                  bool timestamps)
{
    if (use_device_io_)
    {
        return DecodeChunkDevice(lang_token, prompt, timestamps);
    }
    return DecodeChunkHost(lang_token, prompt, timestamps);
}

std::vector<int64_t> WhisperPipeline::DecodeChunkHost(int64_t& lang_token, const std::vector<int64_t>& prompt,
                                                      bool timestamps)
{
    decoded_sum_logprob_ = 0.0;
    decoded_token_count_ = 0;
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
    const auto run = [&](std::vector<int32_t> tokens) -> Ort::Value
    {
        din::common::nvtx_scoped_range range{tokens.size() == 1 ? "decode_step" : "prefill"};
        if (tokens.empty() || tokens.size() > static_cast<size_t>(kMaxPrefillTokens) ||
            pos + static_cast<int64_t>(tokens.size()) > dims_.max_positions)
        {
            throw std::runtime_error("invalid Whisper prefill token count");
        }
        const int64_t sequence = static_cast<int64_t>(tokens.size());
        const std::vector<int64_t> id_shape{1, sequence};
        Ort::Value ids =
            Ort::Value::CreateTensor<int32_t>(cpu, tokens.data(), tokens.size(), id_shape.data(), id_shape.size());
        std::vector<int64_t> write_index(static_cast<size_t>(sequence));
        for (int64_t i = 0; i < sequence; ++i)
        {
            write_index[static_cast<size_t>(i)] = pos + i;
        }
        int64_t nonpad_len = pos + sequence;
        const std::vector<int64_t> write_shape{sequence};
        const std::vector<int64_t> scalar_shape{1};
        Ort::Value write_indices = Ort::Value::CreateTensor<int64_t>(
            cpu, write_index.data(), write_index.size(), write_shape.data(), write_shape.size());
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
        run_options.AddConfigEntry("nv_profile_index", tokens.size() == 1 ? "0" : "1");
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
        pos += sequence;
        return std::move(out_values[index.at("logits")]);
    };

    // Language detection depends on the logits following SOT. Once selected, the
    // remaining prompt is populated in one multi-token prefill invocation.
    std::vector<int32_t> prefix;
    if (!prompt.empty())
    {
        prefix.push_back(static_cast<int32_t>(special_.start_of_prev));
        const size_t keep = std::min<size_t>(prompt.size(), kMaxPrefillTokens - 2);
        for (size_t i = prompt.size() - keep; i < prompt.size(); ++i)
        {
            prefix.push_back(static_cast<int32_t>(prompt[i]));
        }
    }
    prefix.push_back(static_cast<int32_t>(special_.sot));
    Ort::Value logits = run(prefix);
    no_speech_probability_ = LastPositionProbability(logits, special_.no_speech, dims_.vocab);
    if (lang_token < 0)
    {
        lang_token = ResolveLanguageToken(logits);
    }
    std::vector<int32_t> task_prompt{static_cast<int32_t>(lang_token), static_cast<int32_t>(special_.transcribe)};
    if (!timestamps)
    {
        task_prompt.push_back(static_cast<int32_t>(special_.notimestamps));
    }
    logits = run(task_prompt);

    const bool debug = std::getenv("DIN_WHISPER_DEBUG") != nullptr;
    std::vector<int64_t> generated;
    const auto select_timestamp = [&](const Ort::Value& values)
    {
        float selected_logprob = 0.0F;
        const int64_t selected = SelectTimestampToken(values, generated, special_, &selected_logprob);
        decoded_sum_logprob_ += selected_logprob;
        ++decoded_token_count_;
        return selected;
    };
    int64_t next = timestamps ? select_timestamp(logits) : ArgmaxLastPosition(logits, 0, special_.eot + 1);
    while (next != special_.eot && pos < dims_.max_positions)
    {
        generated.push_back(next);
        if (debug && generated.size() <= 8)
        {
            std::cerr << "[debug] pos=" << pos << " token=" << next << std::endl;
        }
        logits = run({static_cast<int32_t>(next)});
        next = timestamps ? select_timestamp(logits) : ArgmaxLastPosition(logits, 0, special_.eot + 1);
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
    encode_seconds_ = 0.0;
    greedy_seconds_ = 0.0;

    const auto transcribe_start = std::chrono::steady_clock::now();

    const size_t total = audio.samples.size();
    const size_t num_chunks = (total + kChunkSamples - 1) / kChunkSamples;
    size_t max_iterations = config_.long_form ? std::numeric_limits<size_t>::max() : std::min<size_t>(num_chunks, 1);
    if (const char* cap = std::getenv("DIN_WHISPER_MAX_CHUNKS"))
    {
        max_iterations = std::min<size_t>(max_iterations, std::strtoul(cap, nullptr, 10));
    }
    result.audio_seconds = static_cast<double>(config_.long_form
                                                   ? total
                                                   : std::min(total, static_cast<size_t>(kChunkSamples))) /
        static_cast<double>(kSampleRate);

    const auto trim = [](std::string value)
    {
        const auto first = value.find_first_not_of(" \t\r\n");
        const auto last = value.find_last_not_of(" \t\r\n");
        return first == std::string::npos ? std::string{} : value.substr(first, last - first + 1);
    };

    std::vector<int64_t> prompt;
    int64_t lang_token = -1;
    size_t seek = 0;
    size_t iteration = 0;
    while (seek < total && iteration < max_iterations)
    {
        const size_t window_start = seek;
        const size_t count = std::min<size_t>(kChunkSamples, total - seek);
        std::vector<float> chunk(kChunkSamples, 0.0F);
        std::copy_n(audio.samples.begin() + seek, count, chunk.begin());

        const auto encode_start = std::chrono::steady_clock::now();
        EncodeChunk(chunk);
        encode_seconds_ += SecondsSince(encode_start);

        const auto greedy_start = std::chrono::steady_clock::now();
        const std::vector<int64_t> tokens = DecodeChunk(lang_token, prompt, config_.long_form);
        if (result.language.empty())
        {
            result.language = tokenizer_->LanguageCode(lang_token);
        }
        greedy_seconds_ += SecondsSince(greedy_start);
        result.model_window_seconds += kChunkSeconds;

        const double average_logprob = decoded_token_count_ > 0
                                           ? decoded_sum_logprob_ / static_cast<double>(decoded_token_count_)
                                           : -INFINITY;
        if (config_.long_form && no_speech_probability_ > 0.6F && average_logprob < -1.0)
        {
            std::cerr << "[window " << (iteration + 1) << " seek=" << std::fixed << std::setprecision(2)
                << static_cast<double>(window_start) / kSampleRate << "s " << result.language
                << "] no speech (p=" << no_speech_probability_ << ")" << std::endl;
            seek += count;
            ++iteration;
            continue;
        }

        if (!config_.long_form)
        {
            WhisperSegment segment;
            segment.start = 0.0;
            segment.end = static_cast<double>(count) / kSampleRate;
            segment.tokens = tokens;
            segment.text = trim(tokenizer_->Decode(tokens));
            result.segments.push_back(std::move(segment));
            seek += count;
        }
        else
        {
            const auto is_timestamp = [&](int64_t token) { return token >= special_.timestamp_first; };
            std::vector<size_t> boundaries;
            for (size_t i = 1; i < tokens.size(); ++i)
            {
                if (is_timestamp(tokens[i - 1]) && is_timestamp(tokens[i]))
                {
                    boundaries.push_back(i);
                }
            }
            const bool single_timestamp_ending = tokens.size() >= 2 && !is_timestamp(tokens[tokens.size() - 2]) &&
                is_timestamp(tokens.back());
            size_t committed = tokens.size();
            size_t seek_advance = count;
            if (!boundaries.empty())
            {
                const size_t segments_before = result.segments.size();
                if (single_timestamp_ending)
                {
                    boundaries.push_back(tokens.size());
                }
                size_t slice_start = 0;
                for (const size_t slice_end : boundaries)
                {
                    if (slice_end > slice_start && is_timestamp(tokens[slice_start]) &&
                        is_timestamp(tokens[slice_end - 1]))
                    {
                        WhisperSegment segment;
                        segment.start = static_cast<double>(window_start) / kSampleRate +
                            static_cast<double>(tokens[slice_start] - special_.timestamp_first) * 0.02;
                        segment.end = static_cast<double>(window_start) / kSampleRate +
                            static_cast<double>(tokens[slice_end - 1] - special_.timestamp_first) * 0.02;
                        segment.tokens.assign(tokens.begin() + slice_start, tokens.begin() + slice_end);
                        segment.text = trim(tokenizer_->Decode(segment.tokens));
                        if (!segment.text.empty())
                        {
                            result.segments.push_back(std::move(segment));
                        }
                    }
                    slice_start = slice_end;
                }
                committed = boundaries.back();
                if (!single_timestamp_ending)
                {
                    const int64_t timestamp_position = tokens[committed - 1] - special_.timestamp_first;
                    seek_advance = static_cast<size_t>(std::max<int64_t>(timestamp_position, 1)) * 320;
                    seek_advance = std::min(seek_advance, count);
                }
                if (result.segments.size() == segments_before)
                {
                    // Empty timestamp pairs carry no text to condition on. Move
                    // through silence in 0.5 s increments instead of re-encoding
                    // nearly the same 30 s window every 20 ms.
                    seek_advance = std::max<size_t>(seek_advance, kSampleRate / 2);
                }
            }
            else
            {
                WhisperSegment segment;
                segment.start = static_cast<double>(window_start) / kSampleRate;
                segment.end = segment.start + static_cast<double>(count) / kSampleRate;
                if (!tokens.empty() && is_timestamp(tokens.front()))
                {
                    segment.start += static_cast<double>(tokens.front() - special_.timestamp_first) * 0.02;
                }
                if (!tokens.empty() && is_timestamp(tokens.back()))
                {
                    segment.end = static_cast<double>(window_start) / kSampleRate +
                        static_cast<double>(tokens.back() - special_.timestamp_first) * 0.02;
                }
                segment.tokens = tokens;
                segment.text = trim(tokenizer_->Decode(tokens));
                if (!segment.text.empty())
                {
                    result.segments.push_back(std::move(segment));
                }
            }
            prompt.insert(prompt.end(), tokens.begin(), tokens.begin() + committed);
            seek += std::max<size_t>(seek_advance, 1);
        }

        std::string iteration_text;
        if (!result.segments.empty())
        {
            iteration_text = result.segments.back().text;
        }
        std::cerr << "[window " << (iteration + 1) << " seek=" << std::fixed << std::setprecision(2)
            << static_cast<double>(window_start) / kSampleRate << "s " << result.language << "] "
            << iteration_text << std::endl;
        ++iteration;
    }

    for (const auto& segment : result.segments)
    {
        if (!result.text.empty() && !segment.text.empty())
        {
            result.text += ' ';
        }
        result.text += segment.text;
    }
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
                            const TranscriptionOptions& options) const
{
    if (!result.language.empty())
    {
        stream << "[language: " << result.language << "]\n";
    }
    if (options.timestamps == "segment")
    {
        for (const auto& segment : result.segments)
        {
            stream << '[' << std::fixed << std::setprecision(2) << segment.start << " --> " << segment.end << "] "
                << segment.text << '\n';
        }
    }
    else
    {
        stream << result.text << '\n';
    }
}

}  // namespace din::asr::whisper
