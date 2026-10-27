// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "asr_common.h"
#include "audio.h"
#include "ort_session.h"
#include "tokenizer.h"
#include <onnxruntime_cxx_api.h>

namespace din::asr::parakeet
{

using Audio = din::io::Audio;
using din::asr::common::TranscriptionOptions;

struct Metadata
{
    int64_t blank_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t vocab_size = 0;
    int64_t max_symbols_per_step = 0;
    int64_t decoder_hidden_size = 0;
    int64_t joint_dim = 0;
    int64_t num_decoder_layers = 0;
    int64_t sampling_rate = 16000;
    int64_t hop_length = 160;
    int64_t subsampling_factor = 8;
    int64_t encoder_profile_frames = 65536;
    std::vector<int64_t> durations;

    [[nodiscard]] double FrameRate() const
    {
        return static_cast<double>(hop_length) / static_cast<double>(sampling_rate) *
               static_cast<double>(subsampling_factor);
    }

    static Metadata Load(const std::filesystem::path& path);
};

class Tokenizer : public din::io::Tokenizer
{
public:
    explicit Tokenizer(const std::string& path)
        : din::io::Tokenizer(path, din::io::TokenizerFormat::Json)
    {
    }
};

struct GenerateOutput
{
    std::vector<int64_t> tokens;
    std::vector<int64_t> durations;
    std::vector<int64_t> starts;
};

struct TranscriptionResult
{
    std::string text;
    GenerateOutput generated;
    double audio_seconds = 0.0;
    double transcribe_seconds = 0.0;
    double encode_seconds = 0.0;
    double greedy_seconds = 0.0;
};

struct ParakeetConfig
{
    std::filesystem::path model_dir = "artifacts/parakeet/onnx";
    std::string provider = "trt-rtx";
    std::filesystem::path ep_cache_dir = "artifacts/parakeet/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/parakeet/ep_context";
    int64_t encoder_profile_frames = 65536;
};

class ParakeetPipeline
{
public:
    explicit ParakeetPipeline(ParakeetConfig config);

    [[nodiscard]] TranscriptionResult Transcribe(const Audio& audio);
    [[nodiscard]] TranscriptionResult TranscribeFile(const std::filesystem::path& audio_path);
    void Print(std::ostream& stream, const TranscriptionResult& result, const TranscriptionOptions& options) const;

private:
    std::unique_ptr<din::common::OrtRunner> MakeRunner(const std::filesystem::path& path,
                                                       const din::common::EpContextOptions& ep_context,
                                                       const din::common::ModelProfile& profile = {});

    ParakeetConfig config_;
    Metadata metadata_;
    std::unique_ptr<Tokenizer> tokenizer_;
    Ort::Env env_;
    Ort::SyncStream compute_stream_{nullptr};
    std::unique_ptr<din::common::OrtRunner> preprocessor_;
    std::unique_ptr<din::common::OrtRunner> encoder_;
    std::unique_ptr<din::common::OrtRunner> predict_;
};

}  // namespace din::asr::parakeet
