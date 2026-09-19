// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "whisper.h"
#include "whisper_kernels.h"

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
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvtx_helper.h"
#include <nlohmann/json.hpp>

#ifdef DIN_WHISPER_CUDA
#include <cuda_runtime_api.h>

#include <onnxruntime_run_options_config_keys.h>
#endif

namespace din::asr::whisper
{
namespace fs = std::filesystem;
namespace
{
TimestampFilter MakeTimestampFilter(std::span<const int64_t> generated, const SpecialTokens& special, int maximum)
{
    const int first = static_cast<int>(special.timestamp_first);
    TimestampFilter filter{static_cast<int>(special.eot), first, first, first + maximum};
    const bool last = !generated.empty() && generated.back() >= first;
    const bool previous = generated.size() < 2 || generated[generated.size() - 2] >= first;
    if (last)
    {
        filter.text = previous;
        if (previous || generated.back() >= filter.timestamp_max)
            filter.timestamp_max = -1;
    }
    for (auto it = generated.rbegin(); it != generated.rend(); ++it)
        if (*it >= first)
        {
            filter.timestamp_min = static_cast<int>(*it) + (last && !previous ? 0 : 1);
            break;
        }
    if (generated.empty())
    {
        filter.text = filter.end = false;
        filter.timestamp_max = std::min(filter.timestamp_max, first + 50);
    }
    return filter;
}

std::vector<int32_t> MakePrefix(std::span<const int64_t> prompt, const SpecialTokens& special)
{
    std::vector<int32_t> prefix;
    if (!prompt.empty())
    {
        prefix.push_back(static_cast<int32_t>(special.start_of_prev));
        const auto history = prompt.last(std::min<size_t>(prompt.size(), kMaxPrefillTokens - 2));
        for (int64_t token : history)
            prefix.push_back(static_cast<int32_t>(token));
    }
    prefix.push_back(static_cast<int32_t>(special.sot));
    return prefix;
}

struct TokenSelection
{
    int64_t token;
    double logprob;
};

// Match Whisper's ApplyTimestampRules: timestamps alternate with text, ending
// times strictly increase, and aggregate timestamp probability competes with
// the best text/EOT candidate. generated never includes the previous prompt.
TokenSelection SelectTimestamp(std::vector<float> scores, TimestampFilter filter,
                               std::span<const int64_t> suppressed)
{
    const int eot = filter.eot, timestamp_first = filter.timestamp_first;
    if (eot < 0 || timestamp_first <= eot || timestamp_first >= static_cast<int64_t>(scores.size()))
        throw std::invalid_argument("invalid Whisper timestamp vocabulary");
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const auto mask = [&](int64_t first, int64_t end)
    {
        first = std::clamp<int64_t>(first, 0, scores.size());
        end = std::clamp<int64_t>(end, first, scores.size());
        std::fill(scores.begin() + first, scores.begin() + end, neg_inf);
    };
    mask(eot + 1, timestamp_first);
    if (!filter.text)
        mask(0, eot);
    if (!filter.end)
        scores[eot] = neg_inf;
    mask(timestamp_first, filter.timestamp_min);
    mask(std::max(timestamp_first, filter.timestamp_max + 1), scores.size());
    for (const auto id : suppressed)
        if (id >= 0 && id < static_cast<int64_t>(scores.size()))
            scores[id] = neg_inf;
    const auto ts = scores.begin() + timestamp_first;
    const double max_timestamp_score = *std::max_element(ts, scores.end());
    const double max_text = *std::max_element(scores.begin(), ts);
    if (std::isfinite(max_timestamp_score))
    {
        double sum = 0;
        for (auto it = ts; it != scores.end(); ++it)
            sum += std::exp(double(*it) - max_timestamp_score);
        if (max_timestamp_score + std::log(sum) > max_text)
            mask(0, timestamp_first);
    }
    auto best = std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(*best))
        throw std::runtime_error("Whisper decoder produced no finite candidate");
    double sum = 0;
    for (float score : scores)
        sum += std::exp(double(score) - *best);
    return {best - scores.begin(), -std::log(sum)};
}

struct TimedTokens
{
    double start = 0, end = 0;
    std::vector<int64_t> tokens;
};

// Detect a short phrase repeated at least four times in succession. Used to
// retry a context-conditioned window without history, never to delete speech.
bool IsRepetitive(std::span<const int64_t> tokens, int64_t eot)
{
    std::vector<int64_t> text;
    for (int64_t token : tokens)
        if (token >= 0 && token < eot)
            text.push_back(token);
    for (size_t length = 1; length <= 16; ++length)
    {
        const size_t repeats = std::max<size_t>(4, (12 + length - 1) / length);
        const size_t needed = repeats * length;
        for (size_t begin = 0; begin + needed <= text.size(); ++begin)
        {
            bool equal = true;
            for (size_t i = length; i < needed && equal; ++i)
                equal = text[begin + i] == text[begin + i % length];
            if (equal)
                return true;
        }
    }
    return false;
}
struct WindowResult
{
    std::vector<TimedTokens> segments;
    std::vector<int64_t> committed;
    size_t advance = 0;
};

WindowResult ParseWindow(std::span<const int64_t> tokens, int64_t timestamp_first, int64_t eot, size_t samples)
{
    WindowResult result;
    result.advance = samples;
    if (tokens.empty() || samples == 0)
        return result;
    const double duration = double(samples) / 16000;
    const auto time = [&](int64_t token)
    {
        return std::clamp((token - timestamp_first) * 0.02, 0.0, duration);
    };
    const auto append = [&](size_t first, size_t end, double start, double finish)
    {
        if (finish <= start)
            return;
        TimedTokens segment{start, finish, {}};
        for (size_t i = first; i < end; ++i)
            if (tokens[i] >= 0 && tokens[i] < eot)
                segment.tokens.push_back(tokens[i]);
        if (!segment.tokens.empty())
        {
            result.segments.push_back(std::move(segment));
            result.committed.insert(result.committed.end(), tokens.begin() + first, tokens.begin() + end);
        }
    };
    const bool ended =
        tokens.size() >= 2 && tokens.back() >= timestamp_first && tokens[tokens.size() - 2] < timestamp_first;
    std::vector<size_t> boundaries;
    for (size_t i = 1; i < tokens.size(); ++i)
        if (tokens[i - 1] >= timestamp_first && tokens[i] >= timestamp_first)
            boundaries.push_back(i);
    if (!boundaries.empty())
    {
        if (ended)
            boundaries.push_back(tokens.size());
        size_t first = 0;
        for (size_t end : boundaries)
        {
            if (tokens[first] >= timestamp_first)
                append(first, end, time(tokens[first]), time(tokens[end - 1]));
            first = end;
        }
        if (!ended)
        {
            const int64_t frames = tokens[first - 1] - timestamp_first;
            if (frames > 0)
                result.advance = std::min(samples, static_cast<size_t>(frames) * 320);
        }
    }
    else
    {
        double end = duration;
        for (auto it = tokens.rbegin(); it != tokens.rend(); ++it)
            if (*it > timestamp_first)
            {
                end = time(*it);
                break;
            }
        if (tokens.back() >= timestamp_first)
            end = time(tokens.back());
        append(0, tokens.size(), tokens.front() >= timestamp_first ? time(tokens.front()) : 0.0, end);
    }
    return result;
}
#ifdef DIN_WHISPER_CUDA
void CheckCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct EncoderTiming
{
    cudaStream_t stream;
    cudaEvent_t events[2]{};
    bool drained = false;
    explicit EncoderTiming(cudaStream_t s)
        : stream(s)
    {
        try
        {
            for (auto& event : events)
                CheckCudaStatus(cudaEventCreate(&event));
        }
        catch (...)
        {
            cudaStreamSynchronize(stream);
            for (auto event : events)
                if (event)
                    cudaEventDestroy(event);
            throw;
        }
    }
    ~EncoderTiming()
    {
        // On failure, finish queued writes before buffers can be reused.
        // Normal completion has already observed the final event.
        if (!drained)
            cudaStreamSynchronize(stream);
        for (auto event : events)
            cudaEventDestroy(event);
    }
};
#endif

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
        const auto* row = logits.GetTensorData<Ort::Float16_t>() + offset;
        for (int64_t id = lower; id < upper; ++id)
        {
            const float v = row[id].ToFloat();
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

std::vector<float> LastPositionLogits(const Ort::Value& logits)
{
    const auto info = logits.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const size_t vocab = static_cast<size_t>(shape.at(2));
    const size_t offset = static_cast<size_t>(shape.at(1) - 1) * vocab;
    std::vector<float> row(vocab);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        const auto* source = logits.GetTensorData<Ort::Float16_t>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            row[i] = source[i].ToFloat();
    }
    else
        std::copy_n(logits.GetTensorData<float>() + offset, vocab, row.begin());
    return row;
}

float LastPositionProbability(const Ort::Value& logits, int64_t token)
{
    const auto row = LastPositionLogits(logits);
    const double maximum = *std::max_element(row.begin(), row.end());
    double sum = 0;
    for (float value : row)
        sum += std::exp(double(value) - maximum);
    return static_cast<float>(std::exp(double(row.at(token)) - maximum) / sum);
}

}  // namespace

WhisperPipeline::WhisperPipeline(WhisperConfig config)
    : config_(std::move(config))
    , env_(ORT_LOGGING_LEVEL_WARNING, "din_asr_whisper")
{
    DIN_NVTX_FUNC_RANGE();
    const fs::path model_dir = config_.model_dir;
    if (config_.prefill_block_size != 0 && (config_.prefill_block_size < 2 || config_.prefill_block_size > kHistoryTokens))
        throw std::invalid_argument("prefill_block_size must be 0 or in [2, 220]");
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

    const std::string tag = config_.model_dir.filename().string();
    ModelProfile mel_profile;
    mel_profile.cache_subpath = tag + "_mel";
    mel_profile.enable_cuda_graph = false;
    ModelProfile encoder_profile;
    encoder_profile.cache_subpath = tag + "_encoder";
    encoder_profile.enable_cuda_graph = false;
    encoder_profile.embed_ep_context = false;
    ModelProfile decoder_profile;
    decoder_profile.cache_subpath = tag + "_decoder_dynamic_" + std::to_string(config_.prefill_block_size);
    decoder_profile.min_shapes = decoder_profile.opt_shapes = "input_ids:1x1,write_indices:1";
    const auto max_sequence = std::to_string(std::max(1, config_.prefill_block_size));
    decoder_profile.max_shapes = "input_ids:1x" + max_sequence + ",write_indices:" + max_sequence;
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
    // Older Transformers writes added_tokens.json; version 5 uses tokenizer.json.
    // vocab.json may also contain the special tokens. Load all three formats.
    std::ifstream vocab_stream(vocab_path);
    nlohmann::json tokens = nlohmann::json::parse(vocab_stream);
    if (fs::exists(model_dir / "tokenizer.json"))
    {
        std::ifstream stream(model_dir / "tokenizer.json");
        const auto data = nlohmann::json::parse(stream);
        if (data.contains("added_tokens"))
            for (const auto& token : data.at("added_tokens"))
                tokens[token.at("content").get<std::string>()] = token.at("id");
    }
    if (fs::exists(added))
    {
        std::ifstream stream(added);
        tokens.update(nlohmann::json::parse(stream));
    }
    {
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
    if (dims_.max_positions < 8 || special_.timestamp_first != special_.notimestamps + 1 ||
        special_.timestamp_first + 1500 >= dims_.vocab)
        throw std::runtime_error("Whisper tokenizer and exported model vocabulary do not match");
    const fs::path generation_path = model_dir / "generation_config.json";
    if (fs::exists(generation_path))
    {
        std::ifstream stream(generation_path);
        const auto generation = nlohmann::json::parse(stream);
        if (generation.contains("suppress_tokens") && !generation.at("suppress_tokens").is_null())
            suppressed_tokens_ = generation.at("suppress_tokens").get<std::vector<int64_t>>();
    }
    else
        throw std::runtime_error("Long-form decoding requires generation_config.json from the source model beside the "
                                 "ONNX files (no graph re-export needed)");

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
        WarmupDecoder();
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
    auto context = ep_context;
    // Profiles of the same graph produce identical engine sidecar names.
    // Keep their compiled caches separate without duplicating the source ONNX.
    if (path.filename() == "decoder.onnx" && !context.output_dir.empty())
        context.output_dir = (fs::path(context.output_dir) / profile.cache_subpath).string();
    return std::make_unique<OrtRunner>(env_, path.string(), config_.provider, config_.ep_cache_dir.string(), context,
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
    if (use_device_io_)
        run_options.SetSyncStream(compute_stream_);
#ifdef DIN_WHISPER_CUDA
    if (use_cuda_sampling_)
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
#endif
    mel_->session.Run(run_options, *mel_binding_);          // samples -> audio_features_ (device)
    encoder_->session.Run(run_options, *encoder_binding_);  // audio_features_ -> cross_kv_ (in place)
}

void WhisperPipeline::SetupDecodePath()
{
    // Persistent decode IO (device path). Self-KV aliases past==present in place;
    // logits stay on-device for GPU sampling and use pinned memory for CPU sampling.
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
    use_cuda_sampling_ = device_is_cuda_ && !config_.disable_cuda_sampling;
    if (use_cuda_sampling_)
    {
        timestamp_suppressed_.emplace(*decoder_, std::vector<int64_t>{dims_.vocab}, true, true);
        std::fill_n(timestamp_suppressed_->HostData(), dims_.vocab, uint8_t{0});
        for (const auto id : suppressed_tokens_)
            if (id >= 0 && id < dims_.vocab)
                timestamp_suppressed_->HostData()[id] = 1;
        timestamp_suppressed_->CopyAsyncToDeviceWithNotification().Sync();
#ifdef DIN_WHISPER_CUDA
        const int64_t workspace_size = whisper_sampling_workspace_size(dims_.vocab);
        timestamp_workspace_.emplace(*decoder_, std::vector<int64_t>{workspace_size}, true, true);
#endif
        timestamp_stats_.emplace(*decoder_, std::vector<int64_t>{2}, true, true);
    }

    if (use_cuda_sampling_)
    {
        // Logits stay on-device for the kernel; only the winning token is copied
        // back through the pinned token buffer.
        dec_logits_ = MakeIoValue(*decoder_, {1, 1, dims_.vocab}, dims_.io_fp16, /*device=*/true);
    }
    else
    {
        // CPU argmax needs the logits on the host; land them in pinned memory.
        auto allocator = decoder_->PinnedAllocator();
        const int64_t shape[] = {1, 1, dims_.vocab};
        dec_logits_ = Ort::Value::CreateTensor(allocator, shape, 3, IoElementType(dims_.io_fp16));
    }

    const auto bind_decoder = [&](Ort::IoBinding& binding, const Ort::Value& ids, const Ort::Value& indices)
    {
        binding.BindInput("input_ids", ids);
        binding.BindInput("write_indices", indices);
        binding.BindInput("nonpad_kv_seqlen", dec_nonpad_->BindingValue());
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            binding.BindInput(dec_past_self_key_in_[i].c_str(), self_kv_[2 * i]);
            binding.BindInput(dec_past_self_value_in_[i].c_str(), self_kv_[2 * i + 1]);
            binding.BindInput(dec_past_cross_key_in_[i].c_str(), cross_kv_[2 * i]);
            binding.BindInput(dec_past_cross_value_in_[i].c_str(), cross_kv_[2 * i + 1]);
            binding.BindOutput(dec_present_self_key_out_[i].c_str(), self_kv_[2 * i]);
            binding.BindOutput(dec_present_self_value_out_[i].c_str(), self_kv_[2 * i + 1]);
        }
        binding.BindOutput("logits", dec_logits_);
    };
    decoder_binding_.emplace(decoder_->session);
    bind_decoder(*decoder_binding_, dec_input_ids_->BindingValue(), dec_write_idx_->BindingValue());
    if (config_.prefill_block_size && InputInfo(decoder_->session, "input_ids").shape.at(1) < 0)
    {
        history_prompt_.emplace(*decoder_, std::vector<int64_t>{kMaxPrefillTokens}, true, true);
        history_ids_.emplace(*decoder_, std::vector<int64_t>{1, config_.prefill_block_size}, true, true);
        history_indices_.emplace(*decoder_, std::vector<int64_t>{config_.prefill_block_size}, true, true);
        history_binding_.emplace(decoder_->session);
        bind_decoder(*history_binding_, history_ids_->BindingValue(), history_indices_->BindingValue());
    }
    if (std::getenv("DIN_WHISPER_DEBUG"))
    {
        std::cerr << "[debug] greedy argmax: " << (use_cuda_sampling_ ? "cuda kernel" : "cpu (pinned logits)")
                  << std::endl;
    }
}

void WhisperPipeline::WarmupDecoder()
{
#ifdef DIN_WHISPER_CUDA
    if (!device_is_cuda_)
        return;
    din::common::nvtx_scoped_range range{"decoder_warmup"};
    auto stream = reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
    ZeroSelfKv();
    for (auto& kv : cross_kv_)
    {
        const size_t bytes = kv.GetTensorTypeAndShapeInfo().GetElementCount() * (dims_.io_fp16 ? 2 : 4);
        CheckCudaStatus(cudaMemsetAsync(kv.GetTensorMutableRawData(), 0, bytes, stream));
    }
    Ort::RunOptions options;
    options.SetSyncStream(compute_stream_);
    // Exercise both shapes and the return to single-token generation before
    // the first recording. Synchronous runs keep staging buffers safe to reuse.
    const auto run = [&](bool history)
    {
        auto& ids = history ? *history_ids_ : *dec_input_ids_;
        auto& indices = history ? *history_indices_ : *dec_write_idx_;
        const int count = history ? config_.prefill_block_size : 1;
        std::fill_n(ids.HostData(), count, static_cast<int32_t>(special_.sot));
        for (int i = 0; i < count; ++i)
            indices.HostData()[i] = i;
        dec_nonpad_->HostData()[0] = count;
        ids.CopyAsyncToDevice();
        indices.CopyAsyncToDevice();
        dec_nonpad_->CopyAsyncToDevice();
        decoder_->session.Run(options, history ? *history_binding_ : *decoder_binding_);
    };
    run(false);
    if (history_binding_)
    {
        run(true);
        run(false);
    }
    ZeroSelfKv();
    CheckCudaStatus(cudaStreamSynchronize(stream));
#endif
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
            CheckCudaStatus(cudaMemsetAsync(kv.GetTensorMutableRawData(), 0, bytes, stream));
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
        const void* logits_dev = dims_.io_fp16 ? static_cast<const void*>(logits.GetTensorData<Ort::Float16_t>() + row)
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

int32_t WhisperPipeline::SelectTimestampHost(const Ort::Value& logits, const std::vector<int64_t>& generated)
{
    din::common::nvtx_scoped_range range{"timestamp_selection"};
    const auto selected = SelectTimestamp(LastPositionLogits(logits),
                                         MakeTimestampFilter(generated, special_, max_timestamp_), suppressed_tokens_);
    decoded_sum_logprob_ += static_cast<float>(selected.logprob);
    ++decoded_token_count_;
    return static_cast<int32_t>(selected.token);
}

void WhisperPipeline::SelectTimestampLogits(const Ort::Value& logits, const std::vector<int64_t>& generated)
{
#ifdef DIN_WHISPER_CUDA
    if (use_cuda_sampling_)
    {
        const auto filter = MakeTimestampFilter(generated, special_, max_timestamp_);
        const void* ptr = dims_.io_fp16 ? static_cast<const void*>(logits.GetTensorData<Ort::Float16_t>())
                                      : static_cast<const void*>(logits.GetTensorData<float>());
        launch_whisper_sample(reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()), ptr, dims_.io_fp16,
                                 dims_.vocab, timestamp_suppressed_->BindingValue().GetTensorData<uint8_t>(), filter,
                                 timestamp_workspace_->BindingValue().GetTensorMutableData<double>(),
                                 dec_input_ids_->BindingValue().GetTensorMutableData<int32_t>(),
                                 timestamp_stats_->BindingValue().GetTensorMutableData<double>());
        CheckCudaStatus(cudaGetLastError());
        ++decoded_token_count_;
        token_ready_notification_ = dec_input_ids_->CopyAsyncToHostWithNotification();
        return;
    }
#endif
    dec_input_ids_->HostData()[0] = SelectTimestampHost(logits, generated);
    token_ready_notification_ = std::move(dec_input_ids_->CopyAsyncToDeviceWithNotification());
    token_ready_notification_.Sync();
}

void WhisperPipeline::PrefillHistoryBlock(std::span<const int32_t> tokens, int64_t position)
{
    din::common::nvtx_scoped_range range{"history_block"};
#ifdef DIN_WHISPER_CUDA
    if (use_cuda_sampling_)
    {
        launch_whisper_history_inputs(reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()),
                                      history_prompt_->BindingValue().GetTensorData<int32_t>(),
                                      config_.prefill_block_size, static_cast<int>(tokens.size()), position,
                                      history_ids_->BindingValue().GetTensorMutableData<int32_t>(),
                                      history_indices_->BindingValue().GetTensorMutableData<int64_t>(),
                                      dec_nonpad_->BindingValue().GetTensorMutableData<int64_t>());
        CheckCudaStatus(cudaGetLastError());
        Ort::RunOptions options;
        options.SetSyncStream(compute_stream_);
        options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        decoder_->session.Run(options, *history_binding_);
        return;
    }
#endif
    std::fill_n(history_ids_->HostData(), config_.prefill_block_size, 0);
    std::copy(tokens.begin(), tokens.end(), history_ids_->HostData());
    for (int i = 0; i < config_.prefill_block_size; ++i)
        history_indices_->HostData()[i] = position + i;
    dec_nonpad_->HostData()[0] = position + tokens.size();
    history_ids_->CopyAsyncToDevice();
    history_indices_->CopyAsyncToDevice();
    auto uploaded = dec_nonpad_->CopyAsyncToDeviceWithNotification();
    Ort::RunOptions options;
    options.SetSyncStream(compute_stream_);
    decoder_->session.Run(options, *history_binding_);
    uploaded.Sync();
}

std::vector<int64_t> WhisperPipeline::DecodeChunkDevice(int64_t& lang_token, const std::vector<int64_t>& prompt)
{
    ZeroSelfKv();
    if (use_cuda_sampling_)
    {
        std::fill_n(timestamp_stats_->HostData(), 2, 0.0);
        timestamp_stats_->CopyAsyncToDevice();
    }
    int64_t pos = 0;
    const auto step = [&](int32_t token, bool update_token = true)
    {
        din::common::nvtx_scoped_range range{"decode_step"};
#ifdef DIN_WHISPER_CUDA
        if (use_cuda_sampling_)
        {
            // Device-written inputs avoid reusing an in-flight pinned staging
            // buffer. Stream order supplies all decoder/selection dependencies.
            launch_whisper_inputs(reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()), token, update_token, pos,
                                  dec_input_ids_->BindingValue().GetTensorMutableData<int32_t>(),
                                  dec_write_idx_->BindingValue().GetTensorMutableData<int64_t>(),
                                  dec_nonpad_->BindingValue().GetTensorMutableData<int64_t>());
        }
        else
#endif
        {
            if (update_token)
            {
                dec_input_ids_->HostData()[0] = token;
                dec_input_ids_->CopyAsyncToDevice();
            }
            dec_write_idx_->HostData()[0] = pos;
            dec_write_idx_->CopyAsyncToDevice();
            dec_nonpad_->HostData()[0] = pos + 1;
            dec_nonpad_->CopyAsyncToDevice();
        }
        Ort::RunOptions run_options;
        run_options.SetSyncStream(compute_stream_);
#ifdef DIN_WHISPER_CUDA
        if (use_cuda_sampling_)
        {
            // Selection and subsequent inference use the same stream. Let Run
            // enqueue asynchronously; CPU sampling keeps the default wait so
            // its pinned logits are ready.
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        }
#endif
        decoder_->session.Run(run_options, *decoder_binding_);
        ++pos;
    };

    // The same ONNX graph uses a fixed history bucket or a single-token profile.
    const auto prefill = [&](const std::vector<int32_t>& tokens, bool history)
    {
        din::common::nvtx_scoped_range range{"prefill"};
        if (tokens.empty() || pos + static_cast<int64_t>(tokens.size()) > dims_.max_positions)
            throw std::runtime_error("invalid Whisper prefill token count");
        size_t offset = 0;
        if (history_binding_ && history && use_cuda_sampling_)
        {
            std::fill_n(history_prompt_->HostData(), kMaxPrefillTokens, 0);
            std::copy(tokens.begin(), tokens.end(), history_prompt_->HostData());
            history_prompt_->CopyAsyncToDevice();
        }
        // Leave the last token (SOT) to the standard decoder, which supplies
        // language/no-speech logits. All earlier known tokens only need KV.
        if (history_binding_ && history)
            while (offset + 1 < tokens.size())
            {
                const size_t count = std::min<size_t>(config_.prefill_block_size, tokens.size() - offset - 1);
                // Pad the final block. Nonpad masks the extra keys; subsequent
                // decoder steps overwrite those unused future cache slots.
                PrefillHistoryBlock(std::span<const int32_t>(tokens).subspan(offset, count), pos);
                pos += count;
                offset += count;
            }
        for (; offset < tokens.size(); ++offset)
            step(tokens[offset]);
        if (history)
        {
#ifdef DIN_WHISPER_CUDA
            if (use_cuda_sampling_)
            {
                const void* ptr = dims_.io_fp16 ? static_cast<const void*>(dec_logits_.GetTensorData<Ort::Float16_t>())
                                                : static_cast<const void*>(dec_logits_.GetTensorData<float>());
                launch_whisper_sample(reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()), ptr,
                                      dims_.io_fp16, dims_.vocab, nullptr, {},
                                      timestamp_workspace_->BindingValue().GetTensorMutableData<double>(), nullptr,
                                      timestamp_stats_->BindingValue().GetTensorMutableData<double>(), special_.no_speech);
            }
            else
#endif
                no_speech_probability_ = LastPositionProbability(dec_logits_, special_.no_speech);
        }
        // A known language needs no token readback.
#ifdef DIN_WHISPER_CUDA
        if (use_cuda_sampling_ && history && (lang_token >= 0 || config_.lang_id != "auto"))
            return lang_token >= 0 ? lang_token : ResolveLanguageToken(dec_logits_);
#endif
        if (history)
            ArgmaxLogits(dec_logits_, special_.lang_first, special_.lang_last + 1);
        else
            SelectTimestampLogits(dec_logits_, {});
        token_ready_notification_.Sync();
        if (dec_input_ids_->HostData()[0] < 0)
            throw std::runtime_error("Whisper decoder produced no finite candidate");
        return static_cast<int64_t>(dec_input_ids_->HostData()[0]);
    };

    const auto prefix = MakePrefix(prompt, special_);
    const int64_t detected = prefill(prefix, true);
    if (lang_token < 0)
    {
        lang_token = config_.lang_id == "auto" ? detected : ResolveLanguageToken(dec_logits_);
    }

    std::vector<int32_t> task_prompt{static_cast<int32_t>(lang_token), static_cast<int32_t>(special_.transcribe)};

    std::vector<int64_t> generated;
    int64_t next = prefill(task_prompt, false);
    while (next != special_.eot && pos < dims_.max_positions)
    {
        generated.push_back(next);
        step(static_cast<int32_t>(next), !use_cuda_sampling_);
        SelectTimestampLogits(dec_logits_, generated);
        token_ready_notification_.Sync();
        next = dec_input_ids_->HostData()[0];
        if (next < 0)
            throw std::runtime_error("Whisper decoder produced no finite candidate");
    }
    if (use_cuda_sampling_)
    {
        timestamp_stats_->CopyAsyncToHostWithNotification().Sync();
        decoded_sum_logprob_ = timestamp_stats_->HostData()[0];
        no_speech_probability_ = static_cast<float>(timestamp_stats_->HostData()[1]);
    }
    return generated;
}

std::vector<int64_t> WhisperPipeline::DecodeChunk(int64_t& lang_token, const std::vector<int64_t>& prompt)
{
    decoded_sum_logprob_ = 0.0;
    decoded_token_count_ = 0;
    if (use_device_io_)
    {
        return DecodeChunkDevice(lang_token, prompt);
    }
    return DecodeChunkHost(lang_token, prompt);
}

std::vector<int64_t> WhisperPipeline::DecodeChunkHost(int64_t& lang_token, const std::vector<int64_t>& prompt)
{
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Fixed-capacity self-KV cache, zero-initialized (stale slots beyond the write
    // position are masked by nonpad_kv_seqlen, but zeroing avoids NaN poisoning).
    std::vector<Ort::Value> self_kv;
    self_kv.reserve(2 * dims_.num_layers);
    for (int i = 0; i < 2 * dims_.num_layers; ++i)
    {
        self_kv.push_back(MakeSelfKv(*decoder_, dims_, false));
    }
    for (auto& kv : self_kv)
        std::memset(kv.GetTensorMutableRawData(), 0, SelfKvBytes(dims_));

    int64_t pos = 0;
    const auto run = [&](int32_t token) -> Ort::Value
    {
        din::common::nvtx_scoped_range range{"decode_step"};
        if (pos >= dims_.max_positions)
            throw std::runtime_error("Whisper decoder cache is full");
        int64_t nonpad_len = pos + 1;
        const std::vector<int64_t> id_shape{1, 1}, scalar_shape{1};
        auto ids = Ort::Value::CreateTensor<int32_t>(cpu, &token, 1, id_shape.data(), id_shape.size());
        auto write_indices = Ort::Value::CreateTensor<int64_t>(cpu, &pos, 1, scalar_shape.data(), scalar_shape.size());
        auto nonpad = Ort::Value::CreateTensor<int64_t>(cpu, &nonpad_len, 1, scalar_shape.data(), scalar_shape.size());

        Ort::IoBinding binding(decoder_->session);
        binding.BindInput("input_ids", ids);
        binding.BindInput("write_indices", write_indices);
        binding.BindInput("nonpad_kv_seqlen", nonpad);
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            binding.BindInput(dec_past_self_key_in_[i].c_str(), self_kv[2 * i]);
            binding.BindInput(dec_past_self_value_in_[i].c_str(), self_kv[2 * i + 1]);
            binding.BindInput(dec_past_cross_key_in_[i].c_str(), cross_kv_[2 * i]);
            binding.BindInput(dec_past_cross_value_in_[i].c_str(), cross_kv_[2 * i + 1]);
        }
        binding.BindOutput("logits", cpu);  // logits always read on host for argmax
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            binding.BindOutput(dec_present_self_key_out_[i].c_str(), cpu);
            binding.BindOutput(dec_present_self_value_out_[i].c_str(), cpu);
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
        for (int i = 0; i < dims_.num_layers; ++i)
        {
            self_kv[2 * i] = std::move(out_values[index.at(dec_present_self_key_out_[i])]);
            self_kv[2 * i + 1] = std::move(out_values[index.at(dec_present_self_value_out_[i])]);
        }
        ++pos;
        return std::move(out_values[index.at("logits")]);
    };

    const auto prefill = [&](const std::vector<int32_t>& tokens)
    {
        din::common::nvtx_scoped_range range{"prefill"};
        Ort::Value logits{nullptr};
        for (int32_t token : tokens)
            logits = run(token);
        return logits;
    };

    // Language detection depends on the logits following SOT. Once selected, the
    // remaining prompt is populated using the same single-token graph.
    const auto prefix = MakePrefix(prompt, special_);
    Ort::Value logits = prefill(prefix);
    no_speech_probability_ = LastPositionProbability(logits, special_.no_speech);
    if (lang_token < 0)
    {
        lang_token = ResolveLanguageToken(logits);
    }
    std::vector<int32_t> task_prompt{static_cast<int32_t>(lang_token), static_cast<int32_t>(special_.transcribe)};
    logits = prefill(task_prompt);

    const bool debug = std::getenv("DIN_WHISPER_DEBUG") != nullptr;
    std::vector<int64_t> generated;
    int64_t next = SelectTimestampHost(logits, generated);
    while (next != special_.eot && pos < dims_.max_positions)
    {
        generated.push_back(next);
        if (debug && generated.size() <= 8)
        {
            std::cerr << "[debug] pos=" << pos << " token=" << next << std::endl;
        }
        logits = run(static_cast<int32_t>(next));
        next = SelectTimestampHost(logits, generated);
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
        throw std::runtime_error("expected 16 kHz audio");
    if (std::any_of(audio.samples.begin(), audio.samples.end(),
                    [](float x)
                    {
                        return !std::isfinite(x);
                    }))
        throw std::runtime_error("audio contains non-finite samples");

    TranscriptionResult result;
    encode_seconds_ = greedy_seconds_ = 0.0;
#ifdef DIN_WHISPER_CUDA
    std::optional<EncoderTiming> encode_timing;
    if (use_cuda_sampling_)
    {
        encode_timing.emplace(reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()));
        encode_timing->drained = true;
    }
#endif
    const auto started = std::chrono::steady_clock::now();
    const size_t total = audio.samples.size();
    result.audio_seconds = double(total) / kSampleRate;
    size_t max_windows = std::numeric_limits<size_t>::max();
    if (const char* cap = std::getenv("DIN_WHISPER_MAX_CHUNKS"))
        max_windows = std::strtoul(cap, nullptr, 10);
    const auto trim = [](std::string text)
    {
        const auto first = text.find_first_not_of(" \t\r\n");
        return first == std::string::npos ? std::string{}
                                          : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    };
    std::vector<int64_t> prompt;
    int64_t language = -1;
    size_t seek = 0;
    while (seek < total && result.windows < max_windows)
    {
        din::common::nvtx_scoped_range range{"transcription_window"};
        const size_t count = std::min<size_t>(kChunkSamples, total - seek);
        const double offset = double(seek) / kSampleRate;
        ++result.windows;
        // Exact/quantized digital silence has no transcript. This is not a VAD
        // threshold: quieter-than-PCM16 input is the only signal skipped here.
        const auto first = audio.samples.begin() + seek;
        if (std::all_of(first, first + count,
                        [](float x)
                        {
                            return std::abs(x) < 1e-8F;
                        }))
        {
            seek += count;
            prompt.clear();
            std::cerr << "[window " << result.windows << " seek=" << offset << "s] digital silence\n";
            continue;
        }
        std::vector<float> samples(kChunkSamples, 0.0F);
        std::copy_n(first, count, samples.begin());
        max_timestamp_ = std::min<int>(1500, static_cast<int>((count + 319) / 320));
        const auto encode_start = std::chrono::steady_clock::now();
#ifdef DIN_WHISPER_CUDA
        if (encode_timing)
        {
            encode_timing->drained = false;
            CheckCudaStatus(cudaEventRecord(encode_timing->events[0], encode_timing->stream));
        }
#endif
        EncodeChunk(samples);
        const double encode_host_seconds = SecondsSince(encode_start);
#ifdef DIN_WHISPER_CUDA
        if (encode_timing)
            CheckCudaStatus(cudaEventRecord(encode_timing->events[1], encode_timing->stream));
#endif
        const auto decode_start = std::chrono::steady_clock::now();
        auto tokens = DecodeChunk(language, prompt);
        const auto low_confidence = [&]
        {
            return decoded_token_count_ && decoded_sum_logprob_ / decoded_token_count_ < -1.0;
        };
        bool repetitive = IsRepetitive(tokens, special_.eot);
        if (!prompt.empty() && (repetitive || low_confidence()))
        {
            result.decoded_tokens += tokens.size();
            prompt.clear();
            tokens = DecodeChunk(language, prompt);
            repetitive = IsRepetitive(tokens, special_.eot);
            std::cerr << "[window " << result.windows << "] retried without previous text\n";
        }
#ifdef DIN_WHISPER_CUDA
        if (encode_timing)
        {
            // Decoding has completed on this stream, so both timing events are
            // already complete. Read timing without introducing an encoder wait.
            float milliseconds = 0;
            CheckCudaStatus(cudaEventElapsedTime(&milliseconds, encode_timing->events[0], encode_timing->events[1]));
            encode_timing->drained = true;
            const double encode_seconds = milliseconds / 1000.0;
            encode_seconds_ += encode_seconds;
            greedy_seconds_ += std::max(0.0, SecondsSince(encode_start) - encode_seconds);
        }
        else
#endif
        {
            encode_seconds_ += encode_host_seconds;
            greedy_seconds_ += SecondsSince(decode_start);
        }
        result.decoded_tokens += tokens.size();
        result.model_window_seconds += kChunkSeconds;
        result.language = tokenizer_->LanguageCode(language);
        const double average_logprob = decoded_token_count_ ? decoded_sum_logprob_ / decoded_token_count_ : -INFINITY;
        if (no_speech_probability_ > 0.6F && average_logprob < -1.0)
        {
            seek += count;
            prompt.clear();
            std::cerr << "[window " << result.windows << " seek=" << offset << "s] no speech\n";
            continue;
        }
        auto window = ParseWindow(tokens, special_.timestamp_first, special_.eot, count);
        for (auto& segment : window.segments)
        {
            auto text = trim(tokenizer_->Decode(segment.tokens));
            if (!text.empty())
                result.segments.push_back(
                    {offset + segment.start, offset + segment.end, std::move(text), std::move(segment.tokens)});
        }
        prompt.insert(prompt.end(), window.committed.begin(), window.committed.end());
        const size_t limit = std::min<size_t>(kHistoryTokens, dims_.max_positions / 2 - 4);
        if (prompt.size() > limit)
            prompt.erase(prompt.begin(), prompt.end() - limit);
        if (!config_.condition_on_previous_text || average_logprob < -1.0 || repetitive)
            prompt.clear();
        seek += std::clamp<size_t>(window.advance, 1, count);
        std::cerr << "[window " << result.windows << " seek=" << offset << "s -> " << double(seek) / kSampleRate << "s "
                  << result.language << "] tokens=" << tokens.size() << " avg_logprob=" << average_logprob << '\n';
    }
    for (const auto& segment : result.segments)
    {
        if (!result.text.empty() && !segment.text.empty())
            result.text += ' ';
        result.text += segment.text;
    }
    result.encode_seconds = encode_seconds_;
    result.greedy_seconds = greedy_seconds_;
    result.transcribe_seconds = SecondsSince(started);
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
    if (options.timestamps == "json")
    {
        nlohmann::json segments = nlohmann::json::array();
        for (const auto& segment : result.segments)
            segments.push_back(
                {{"start", segment.start}, {"end", segment.end}, {"text", segment.text}, {"tokens", segment.tokens}});
        stream << nlohmann::json{{"text", result.text},
                                 {"language", result.language},
                                 {"segments", segments},
                                 {"audio_seconds", result.audio_seconds},
                                 {"windows", result.windows},
                                 {"decoded_tokens", result.decoded_tokens},
                                 {"encode_seconds", result.encode_seconds},
                                 {"decode_seconds", result.greedy_seconds},
                                 {"transcribe_seconds", result.transcribe_seconds}}
                      .dump()
               << '\n';
        return;
    }
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
