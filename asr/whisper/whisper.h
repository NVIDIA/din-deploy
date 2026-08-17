// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio.h"
#include "ort_session.h"
#include "tokenizer.h"
#include <onnxruntime_cxx_api.h>

// --- timing helper -------------------------------

inline double SecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

namespace din::asr::whisper
{

using Audio = din::io::Audio;
using din::common::EpContextOptions;
using din::common::ModelProfile;
using din::common::OrtRunner;

struct TranscriptionOptions
{
    std::string timestamps = "none";
};

struct CliSpec
{
    std::string fallback_name;                    // executable name when argv[0] is unavailable
    std::string artifact_label;                   // e.g. "Nemotron" / "Parakeet"
    std::string default_provider;                 // shown in --provider help line
    std::string timestamps_modes;                 // e.g. "none|token"
    std::vector<std::string> allowed_timestamps;  // empty disables validation
    std::vector<std::string> allowed_lang_ids;    // empty disables lang-id validation
};

inline constexpr int kSampleRate = 16000;
inline constexpr int kChunkSeconds = 30;
inline constexpr int kChunkSamples = kSampleRate * kChunkSeconds;  // 480000, mel.onnx input length

struct TranscriptionResult
{
    std::string text;
    std::string language;  // detected/forced language code of the first chunk
    double audio_seconds = 0.0;
    double transcribe_seconds = 0.0;
    double encode_seconds = 0.0;
    double greedy_seconds = 0.0;
};

// Model geometry + IO precision, detected from the ONNX sessions at load time so
// one binary handles every Whisper size (tiny..large) and fp16/fp32 exports.
struct ModelDims
{
    int num_layers = 0;
    int num_heads = 0;
    int head_dim = 0;
    int encoder_frames = 0;  // cross-attention length (1500)
    int max_positions = 0;   // self-KV cache capacity / context length (448)
    int64_t vocab = 0;
    bool io_fp16 = true;  // true: FLOAT16 IO, false: FLOAT (fp32)
};

// Whisper control-token ids. Defaults are the 99-language layout (tiny..medium);
// large-v3 shifts the task tokens, so these are read from added_tokens.json when
// present.
struct SpecialTokens
{
    int64_t sot = 50258;           // <|startoftranscript|>
    int64_t transcribe = 50359;    // <|transcribe|>
    int64_t notimestamps = 50363;  // <|notimestamps|>
    int64_t eot = 50257;           // <|endoftext|>
    int64_t lang_first = 50259;    // <|en|>
    int64_t lang_last = 50357;     // last language tag
};

struct WhisperConfig
{
    std::filesystem::path model_dir = "D:/models/whisper-medium-fp16-trt-rtx";
    std::string provider = "trt-rtx";
    std::filesystem::path ep_cache_dir = "artifacts/whisper/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/whisper/ep_context";
    // Language code (e.g. "en") or "auto" to detect per chunk.
    std::string lang_id = "auto";
    // Force greedy argmax onto the CPU even when the CUDA kernel is available.
    bool disable_cuda_sampling = false;
};

class WhisperPipeline
{
public:
    explicit WhisperPipeline(WhisperConfig config);

    [[nodiscard]] TranscriptionResult Transcribe(const Audio& audio);
    [[nodiscard]] TranscriptionResult TranscribeFile(const std::filesystem::path& audio_path);
    void Print(std::ostream& stream, const TranscriptionResult& result, const TranscriptionOptions& options) const;

private:
    // Allocates the mel/encoder device buffers + IoBindings once (fixed shapes),
    // reused across every chunk.
    void SetupEncodePath();
    // Runs mel.onnx then the encoder for one 30 s window into the persistent
    // cross_kv_ buffers. samples are staged through pinned memory + CopyTensor.
    void EncodeChunk(const std::vector<float>& samples);
    // Picks the language tag id from the logits after <|startoftranscript|>, or
    // resolves a forced --lang-id.
    [[nodiscard]] int64_t ResolveLanguageToken(const Ort::Value& sot_logits) const;
    // Greedily decodes one 30 s window (using cross_kv_); sets `lang_token`.
    std::vector<int64_t> DecodeChunk(int64_t& lang_token);
    // TRT-RTX path: persistent pinned bindings, self-KV aliased in place.
    std::vector<int64_t> DecodeChunkDevice(int64_t& lang_token);
    // CPU path: out-of-place present->past, fresh bindings per step.
    std::vector<int64_t> DecodeChunkHost(int64_t& lang_token);
    void SetupDecodePath();  // allocates the persistent device decode buffers/binding
    void ZeroSelfKv();       // clears the persistent self-KV cache before a chunk
    // Greedy argmax over dec_logits_[lower, upper).
    // output is store inside token_out_
    void ArgmaxLogits(int64_t lower, int64_t upper);

    std::unique_ptr<OrtRunner> MakeRunner(const std::filesystem::path& path, const EpContextOptions& ep_context,
                                          const ModelProfile& profile);
    void DetectModel();  // fills dims_ from the encoder/decoder sessions

    WhisperConfig config_;
    Ort::Env env_;
    Ort::SyncStream compute_stream_{nullptr};
    std::unique_ptr<OrtRunner> mel_;
    std::unique_ptr<OrtRunner> encoder_;
    std::unique_ptr<OrtRunner> decoder_;
    std::unique_ptr<din::io::Tokenizer> tokenizer_;
    ModelDims dims_;
    SpecialTokens special_;
    bool use_device_io_ = false;
    // CUDA compiled in AND the EP device is an NVIDIA GPU, so its device memory is
    // CUDA-addressable (usable for cudaMemset etc.), independent of --cpu-sampling.
    bool device_is_cuda_ = false;
    // Runtime choice: run greedy argmax as a CUDA kernel. device_is_cuda_ and not
    // disabled on the CLI.
    bool use_cuda_sampling_ = false;

    // Persistent mel+encoder IO (fixed shapes; allocated once in SetupEncodePath).
    // samples staged through pinned host memory; everything else stays on-device.
    std::optional<din::common::TensorBuffer<float>> samples_;
    Ort::Value audio_features_{nullptr};  // mel output == encoder input
    Ort::Value hidden_states_{nullptr};   // encoder output (unused, but must bind)
    std::vector<Ort::Value> cross_kv_;    // encoder cross-KV, consumed by the decoder
    std::optional<Ort::IoBinding> mel_binding_;
    std::optional<Ort::IoBinding> encoder_binding_;

    // Persistent decode IO for the TRT-RTX path: fixed shapes, allocated once.
    // Self-KV is aliased (past==present) in place; the three scalar inputs are
    // staged through pinned memory; logits land in pinned host memory.
    std::vector<Ort::Value> self_kv_;
    std::optional<din::common::TensorBuffer<int32_t>> dec_input_ids_;
    std::optional<din::common::TensorBuffer<int64_t>> dec_write_idx_;
    std::optional<din::common::TensorBuffer<int64_t>> dec_nonpad_;
    Ort::Value dec_logits_{nullptr};  // device buffer (CUDA) or pinned host (fallback)
    din::common::NotificationPtr token_ready_notification_{nullptr};
    std::optional<Ort::IoBinding> decoder_binding_;

    // Precomputed, stably-stored ONNX IO names (referenced by IoBinding).
    std::vector<std::string> enc_cross_key_out_;
    std::vector<std::string> enc_cross_value_out_;
    std::vector<std::string> dec_past_self_key_in_;
    std::vector<std::string> dec_past_self_value_in_;
    std::vector<std::string> dec_past_cross_key_in_;
    std::vector<std::string> dec_past_cross_value_in_;
    std::vector<std::string> dec_present_self_key_out_;
    std::vector<std::string> dec_present_self_value_out_;

    double encode_seconds_ = 0.0;
    double greedy_seconds_ = 0.0;
};

}  // namespace din::asr::whisper
