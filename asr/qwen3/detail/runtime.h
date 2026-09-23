// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef DIN_QWEN3_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

#include "audio.h"
#include "nvtx_helper.h"
#include "ort_session.h"
#include "tokenizer.h"
#include <nlohmann/json.hpp>

namespace din::asr::qwen3::detail
{
using Json = nlohmann::json;
using BF16 = Ort::BFloat16_t;
using din::common::OrtRunner;
template <class T>
using Buffer = din::common::TensorBuffer<T>;
class FloatBuffer
{
    using Storage = std::variant<Buffer<float>, Buffer<Ort::Float16_t>, Buffer<BF16>>;
    Storage buffer_;

    static Storage Make(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype,
                        bool disable_uma)
    {
        switch (dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return Storage(std::in_place_type<Buffer<float>>, runner, shape, runner.HasDeviceIo(), disable_uma);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return Storage(std::in_place_type<Buffer<Ort::Float16_t>>, runner, shape, runner.HasDeviceIo(),
                           disable_uma);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return Storage(std::in_place_type<Buffer<BF16>>, runner, shape, runner.HasDeviceIo(), disable_uma);
        default:
            throw std::runtime_error("Unsupported Qwen3 tensor precision");
        }
    }

public:
    FloatBuffer(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype,
                bool disable_uma = false)
        : buffer_(Make(runner, shape, dtype, disable_uma))
    {
    }

    template <class F>
    void WithHost(F&& fill)
    {
        std::visit(
            [&](auto& buffer)
            {
                fill(buffer.HostData());
            },
            buffer_);
    }
    void Fill(float value)
    {
        std::visit(
            [&](auto& buffer)
            {
                using T = std::remove_pointer_t<decltype(buffer.HostData())>;
                buffer.Fill(T(value));
            },
            buffer_);
    }
    Ort::Value& BindingValue()
    {
        return std::visit(
            [](auto& buffer) -> Ort::Value&
            {
                return buffer.BindingValue();
            },
            buffer_);
    }
    void CopyAsyncToDevice()
    {
        std::visit(
            [](auto& buffer)
            {
                buffer.CopyAsyncToDevice();
            },
            buffer_);
    }
    void UploadAndWait()
    {
        std::visit(
            [](auto& buffer)
            {
                buffer.CopyAsyncToDeviceWithNotification().Sync();
            },
            buffer_);
    }
};

inline size_t ElementBytes(const Ort::Value& value)
{
    return value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 2;
}

constexpr int kRate = 16000;

inline int64_t AudioTokens(int64_t frames)
{
    return frames / 100 * 13 + ((frames % 100) + 7) / 8;
}

#ifdef DIN_QWEN3_CUDA
inline bool UsesCudaMemory(const OrtRunner& runner)
{
    if (!runner.HasDeviceIo())
        return false;
    const auto hardware = runner.ep_device.Device();
    return hardware.Type() == OrtHardwareDeviceType_GPU && hardware.VendorId() == 0x10DE;
}
#endif

inline void ZeroTensor(OrtRunner& runner, Ort::Value& value)
{
    const auto bytes = value.GetTensorTypeAndShapeInfo().GetElementCount() * ElementBytes(value);
    if (!runner.HasDeviceIo())
    {
        std::memset(value.GetTensorMutableRawData(), 0, bytes);
        return;
    }
#ifdef DIN_QWEN3_CUDA
    if (UsesCudaMemory(runner))
    {
        const auto status = cudaMemsetAsync(value.GetTensorMutableRawData(), 0, bytes,
                                            reinterpret_cast<cudaStream_t>(runner.compute_stream->GetHandle()));
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        return;
    }
#endif
    std::vector<uint8_t> zeros(bytes, 0);
    const auto info = value.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto source =
        Ort::Value::CreateTensor(memory, zeros.data(), bytes, shape.data(), shape.size(), info.GetElementType());
    Ort::ThrowOnError(runner.env.CopyTensor(source, value, *runner.compute_stream));
    din::common::NotificationPtr done(*runner.compute_stream);
    done.Record();
    done.Sync();
}

inline Json ReadJson(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Missing model asset: " + path.string() + "; re-export the model");
    return Json::parse(file);
}

inline void Append(std::vector<int64_t>& dst, const std::vector<int64_t>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

inline Ort::Value TensorValue(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype)
{
    if (runner.HasDeviceIo())
        return Ort::Value::CreateTensor(runner.DeviceAllocator(), shape.data(), shape.size(), dtype);
    Ort::AllocatorWithDefaultOptions allocator;
    return Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), dtype);
}

struct EncodedAudio
{
    int64_t tokens;
    Ort::Value values;
};

inline void CopyTensor(OrtRunner& runner, Ort::Value& dst, const Ort::Value& src, size_t count, size_t dst_offset = 0,
                       size_t src_offset = 0)
{
    const auto bytes = ElementBytes(dst);
    auto* target = static_cast<char*>(dst.GetTensorMutableRawData()) + dst_offset * bytes;
    const auto* source = static_cast<const char*>(src.GetTensorRawData()) + src_offset * bytes;
    if (!runner.HasDeviceIo())
    {
        std::memcpy(target, source, count * bytes);
        return;
    }
#ifdef DIN_QWEN3_CUDA
    if (UsesCudaMemory(runner))
    {
        const auto status = cudaMemcpyAsync(target, source, count * bytes, cudaMemcpyDeviceToDevice,
                                            reinterpret_cast<cudaStream_t>(runner.compute_stream->GetHandle()));
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        return;
    }
#endif
    const int64_t shape = static_cast<int64_t>(count);
    const auto dtype = dst.GetTensorTypeAndShapeInfo().GetElementType();
    auto from =
        Ort::Value::CreateTensor(src.GetTensorMemoryInfo(), const_cast<char*>(source), count * bytes, &shape, 1, dtype);
    auto to = Ort::Value::CreateTensor(dst.GetTensorMemoryInfo(), target, count * bytes, &shape, 1, dtype);
    Ort::ThrowOnError(runner.env.CopyTensor(from, to, *runner.compute_stream));
}

inline const din::io::Audio& NormalizeAudio(const din::io::Audio& audio, din::io::Audio& normalized)
{
    if (audio.sample_rate != kRate || audio.samples.empty())
        throw std::runtime_error("Expected nonempty mono 16 kHz audio");
    float peak = 0.f;
    for (float sample : audio.samples)
    {
        if (!std::isfinite(sample))
            throw std::runtime_error("Audio contains non-finite samples");
        peak = std::max(peak, std::abs(sample));
    }
    const auto* source = &audio;
    if (peak > 1.f)
    {
        normalized.sample_rate = kRate;
        normalized.samples.resize(audio.samples.size());
        std::transform(audio.samples.begin(), audio.samples.end(), normalized.samples.begin(),
                       [peak](float x)
                       {
                           return x / peak;
                       });
        source = &normalized;
    }
    return *source;
}

struct TextInputs
{
    Buffer<int64_t> ids, positions, logits_index;
    Ort::Value audio;
    FloatBuffer bias;
    Buffer<bool> mask;
    int64_t sequence, hidden, capacity;
    OrtRunner& runner;
    bool mask_uploaded = false;

    TextInputs(OrtRunner& runner, int64_t seq, int64_t width, int64_t keys, ONNXTensorElementDataType dtype)
        : ids(runner, {1, seq}, runner.HasDeviceIo())
        , positions(runner, {1, seq}, runner.HasDeviceIo())
        , logits_index(runner, {1}, runner.HasDeviceIo())
        , audio(TensorValue(runner, {1, seq, width}, dtype))
        , bias(runner, {1, 1, seq, keys}, dtype)
        , mask(runner, {1, seq, 1}, runner.HasDeviceIo())
        , sequence(seq)
        , hidden(width)
        , capacity(keys)
        , runner(runner)
    {
        ZeroTensor(runner, audio);
        mask.Fill(false);
    }

    void Fill(std::span<const int64_t> tokens, int64_t start, int64_t audio_id, const EncodedAudio* embeddings,
              int64_t audio_offset = 0)
    {
        if (tokens.empty() || tokens.size() > static_cast<size_t>(sequence) || start + sequence > capacity)
            throw std::runtime_error("Input exceeds the exported context capacity");
        int64_t audio_pos = audio_offset;
        bool mask_changed = false;
        for (int64_t i = 0; i < sequence; ++i)
        {
            ids.HostData()[i] = i < static_cast<int64_t>(tokens.size()) ? tokens[i] : 0;
            positions.HostData()[i] = start + i;
            const bool is_audio = embeddings && i < static_cast<int64_t>(tokens.size()) && tokens[i] == audio_id;
            mask_changed |= mask.HostData()[i] != is_audio;
            mask.HostData()[i] = is_audio;
            if (is_audio)
            {
                if (audio_pos >= embeddings->tokens)
                    throw std::runtime_error("Too many audio placeholders");
                ++audio_pos;
            }
        }
        bias.WithHost(
            [&](auto* data)
            {
                using T = std::remove_pointer_t<decltype(data)>;
                for (int64_t i = 0; i < sequence; ++i)
                {
                    const int64_t visible = start + i + 1;
                    std::fill_n(data + i * capacity, visible, T(0.f));
                    std::fill_n(data + i * capacity + visible, capacity - visible, T(-1e4f));
                }
            });
        logits_index.HostData()[0] = tokens.size() - 1;
        logits_index.CopyAsyncToDevice();
        ids.CopyAsyncToDevice();
        positions.CopyAsyncToDevice();
        // Audio placeholders form contiguous runs; assemble embeddings directly on
        // the shared stream instead of downloading each encoder window to the CPU.
        audio_pos = audio_offset;
        for (int64_t i = 0; embeddings && i < static_cast<int64_t>(tokens.size());)
        {
            if (tokens[i] != audio_id)
            {
                ++i;
                continue;
            }
            const int64_t begin = i;
            while (i < static_cast<int64_t>(tokens.size()) && tokens[i] == audio_id)
                ++i;
            CopyTensor(runner, audio, embeddings->values, (i - begin) * hidden, begin * hidden, audio_pos * hidden);
            audio_pos += i - begin;
        }
        if (!mask_uploaded || mask_changed)
        {
            mask.CopyAsyncToDevice();
            mask_uploaded = true;
        }
        bias.CopyAsyncToDevice();
    }

    void Bind(Ort::IoBinding& binding, bool decoder = false)
    {
        if (decoder)
            binding.BindInput("logits_index", logits_index.BindingValue());
        binding.BindInput("input_ids", ids.BindingValue());
        binding.BindInput("position_ids", positions.BindingValue());
        binding.BindInput("audio_embeddings", audio);
        binding.BindInput("audio_mask", mask.BindingValue());
        binding.BindInput("attention_bias", bias.BindingValue());
    }
};
inline std::string LowerLanguage(const std::string& language)
{
    const auto first = language.find_first_not_of(" \r\n\t");
    auto result = first == std::string::npos ? std::string{}
                                             : language.substr(first, language.find_last_not_of(" \r\n\t") - first + 1);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });
    return result;
}

struct Runtime
{
    std::string provider;
    std::filesystem::path ep_cache_dir, ep_context_dir;
    din::common::ProgressCallback progress;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "din_asr_qwen3"};
    Ort::SyncStream stream{nullptr};
    std::unique_ptr<OrtRunner> mel;

    Runtime(const std::string& execution_provider, const std::filesystem::path& cache,
            const std::filesystem::path& context, din::common::ProgressCallback callback)
        : provider(execution_provider)
        , ep_cache_dir(cache)
        , ep_context_dir(context)
        , progress(std::move(callback))
    {
        if (provider == "trt-rtx")
        {
            din::common::RegisterTensorRTRTXProvider(env);
            stream = din::common::CreateTensorRTRTXComputeStream(env);
        }
    }
    void LoadMel(const std::filesystem::path& dir)
    {
        mel = Runner(dir, "mel", "samples:1x8000", "samples:1x2880000", "samples:1x19280000");
    }
    std::unique_ptr<OrtRunner> Runner(const std::filesystem::path& dir, const std::string& name, const std::string& min,
                                      const std::string& opt, const std::string& max, const std::string& variant = "")
    {
        din::common::ModelProfile profile;
        profile.min_shapes = min;
        profile.opt_shapes = opt;
        profile.max_shapes = max;
        const auto metadata = ReadJson(dir / "metadata.json");
        const auto identity = metadata.contains("graphs") && metadata["graphs"].contains(name)
                                  ? metadata["graphs"][name].get<std::string>().substr(0, 16)
                                  : dir.filename().string();
        profile.cache_subpath = "qwen3_" + name + "_" + identity + "_fixed" + variant;
        if (name == "decoder")
            profile.cache_subpath += "_kv" + metadata["cache_capacity"].dump();
        if (name == "aligner")
            profile.cache_subpath += "_bins";
        profile.enable_cuda_graph = name != "aligner";
        if (!profile.enable_cuda_graph)
        {
            // Alignment runs once per chunk with changing shapes. Keep graphs
            // for repeated encoder windows and fixed-shape AR steps.
            profile.extra_ep_options.emplace_back("enable_cuda_graph", "0");
            profile.cache_subpath += "_no_graph";
        }
        profile.embed_ep_context = false;
        // ORT names external engines by graph, so different profiles need separate directories.
        return std::make_unique<OrtRunner>(
            env, (dir / (name + ".onnx")).string(), provider, ep_cache_dir.string(),
            din::common::EpContextOptions{(ep_context_dir / profile.cache_subpath).string(), progress}, profile,
            stream ? &stream : nullptr);
    }

    std::vector<float> Features(std::span<const float> audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.mel"};
        const int64_t samples = std::max<int64_t>(8000, audio.size());
        Buffer<float> input(*mel, {1, samples}, mel->HasDeviceIo()),
            output(*mel, {1, 128, samples / 160}, mel->HasDeviceIo());
        input.Fill(0.f);
        std::copy(audio.begin(), audio.end(), input.HostData());
        input.CopyAsyncToDevice();
        Ort::IoBinding binding(mel->session);
        binding.BindInput("samples", input.BindingValue());
        binding.BindOutput("features", output.BindingValue());
        mel->session.Run(Ort::RunOptions{}, binding);
        output.CopyAsyncToHostWithNotification().Sync();
        return {output.HostData(), output.HostData() + 128 * (samples / 160)};
    }
};
struct AudioModel
{
    Json metadata, native;
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    std::unique_ptr<din::io::Tokenizer> tokenizer;
    std::unique_ptr<OrtRunner> encoder;
    int64_t hidden = 0, audio_id = 0, window = 0;

    struct EncoderBuffers
    {
        int64_t frames, tokens;
        FloatBuffer mel, bias;
        Buffer<int64_t> indices;
        Ort::Value output;
        Ort::IoBinding binding;

        EncoderBuffers(OrtRunner& runner, int64_t count, int64_t hidden, ONNXTensorElementDataType dtype)
            : frames(count)
            , tokens(AudioTokens(count))
            , mel(runner, {(count + 99) / 100, 128, 100}, dtype, true)
            , bias(runner, {1, 1, tokens, tokens}, dtype, true)
            , indices(runner, {tokens}, runner.HasDeviceIo())
            , output(TensorValue(runner, {tokens, hidden}, dtype))
            , binding(runner.session)
        {
            // A call contains exactly one independent HF encoder window.
            bias.Fill(0.f);
            for (int64_t i = 0; i < tokens; ++i)
                indices.HostData()[i] = i;
            bias.CopyAsyncToDevice();
            indices.CopyAsyncToDevice();
            binding.BindInput("mel_chunks", mel.BindingValue());
            binding.BindInput("valid_indices", indices.BindingValue());
            binding.BindInput("attention_bias", bias.BindingValue());
            binding.BindOutput("audio_embeddings", output);
        }
    };

    // Pad the last encoder window so all audio lengths reuse one GPU shape.
    std::unique_ptr<EncoderBuffers> full_window;

    EncodedAudio Encode(const std::vector<float>& features, int64_t frames)
    {
        const int64_t tokens = AudioTokens(frames);
        EncodedAudio result{tokens, TensorValue(*encoder, {tokens, hidden}, dtype)};
        Ort::RunOptions options;
        if (encoder->HasDeviceIo())
            options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        int64_t token_offset = 0;
        for (int64_t offset = 0; offset < frames; offset += window)
        {
            din::common::nvtx_scoped_range range{"qwen3.encoder_window"};
            const auto count = std::min(window, frames - offset);
            if (!full_window)
                full_window = std::make_unique<EncoderBuffers>(*encoder, window, hidden, dtype);
            auto& input = *full_window;
            const auto valid_tokens = AudioTokens(count);
            input.bias.WithHost(
                [&](auto* data)
                {
                    using T = std::remove_pointer_t<decltype(data)>;
                    for (int64_t i = 0; i < input.tokens; ++i)
                    {
                        std::fill_n(data + i * input.tokens, valid_tokens, T(0.f));
                        std::fill_n(data + i * input.tokens + valid_tokens, input.tokens - valid_tokens, T(-1e4f));
                    }
                });
            input.bias.CopyAsyncToDevice();
            input.mel.Fill(0.f);
            input.mel.WithHost(
                [&](auto* data)
                {
                    using T = std::remove_pointer_t<decltype(data)>;
                    for (int64_t c = 0; c < (count + 99) / 100; ++c)
                        for (int64_t m = 0; m < 128; ++m)
                            for (int64_t f = 0; f < 100 && c * 100 + f < count; ++f)
                                data[(c * 128 + m) * 100 + f] = T(features[m * frames + offset + c * 100 + f]);
                });
            // Complete only the upload before reusing host staging. The next
            // window's CPU preparation can overlap this window's inference.
            input.mel.UploadAndWait();
            encoder->session.Run(options, input.binding);
            CopyTensor(*encoder, result.values, input.output, valid_tokens * hidden, token_offset * hidden);
            token_offset += valid_tokens;
        }
        return result;
    }
    AudioModel(Runtime& runtime, const std::filesystem::path& dir, const std::string& task)
    {
        metadata = ReadJson(dir / "metadata.json");
        native = ReadJson(dir / "native.json");
        const auto& meta = metadata;
        if (meta.at("format_version") != (task == "asr" ? 3 : 2) || meta.at("task") != task)
            throw std::runtime_error("Re-export Qwen3 for the unified in-place decoder");
        const auto precision = meta.at("dtype").get<std::string>();
        if (precision == "bfloat16")
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        else if (precision == "float16")
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        else if (precision == "float32")
            dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        else
            throw std::runtime_error("Unsupported Qwen3 export precision: " + precision);
        if (meta["audio_config"]["num_mel_bins"] != 128 || meta["audio_config"]["n_window"] != 50 ||
            meta["audio_config"]["n_window_infer"] != 800)
            throw std::runtime_error("Unsupported audio geometry");
        window = meta["audio_config"]["n_window_infer"];
        hidden = meta["text_config"]["hidden_size"];
        audio_id = meta["audio_token_id"];
        tokenizer = std::make_unique<din::io::Tokenizer>((dir / "processor/tokenizer.json").string(),
                                                         din::io::TokenizerFormat::ByteBpeJson);
        auto enc_shapes = [](int chunks, int tokens)
        {
            return "mel_chunks:" + std::to_string(chunks) + "x128x100,valid_indices:" + std::to_string(tokens) +
                   ",attention_bias:1x1x" + std::to_string(tokens) + "x" + std::to_string(tokens);
        };
        encoder = runtime.Runner(dir, "encoder", enc_shapes(8, 104), enc_shapes(8, 104), enc_shapes(8, 104));
    }
};
inline std::string TextShape(int64_t seq, int64_t hidden, int64_t keys)
{
    const auto s = std::to_string(seq);
    return "input_ids:1x" + s + ",audio_embeddings:1x" + s + "x" + std::to_string(hidden) + ",audio_mask:1x" + s +
           "x1,position_ids:1x" + s + ",attention_bias:1x1x" + s + "x" + std::to_string(keys);
}

}  // namespace din::asr::qwen3::detail
