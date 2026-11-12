// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr_common.h"
#include "audio.h"
#include "ort_session.h"
#include "tokenizer.h"
#include <onnxruntime_cxx_api.h>

namespace din::asr::nemotron
{

using Audio = din::io::Audio;
using din::asr::common::TranscriptionOptions;

struct Metadata
{
    int64_t blank_token_id = 0;
    int64_t max_symbols_per_step = 10;
    int64_t decoder_hidden_size = 640;
    int64_t joint_dim = 1024;
    int64_t num_decoder_layers = 2;
    int64_t sampling_rate = 16000;
    int64_t hop_length = 160;
    int64_t num_mel_bins = 128;
    int64_t subsampling_factor = 8;
    int64_t preprocessor_profile_frames = 65536;
    int64_t max_mel_frames = 25;
    int64_t chunk_size = 4;
    int64_t encoder_step_input_frames = 25;
    int64_t chunk_feature_frames_first = 25;
    int64_t chunk_feature_frames_next = 32;
    int64_t shift_feature_frames_first = 25;
    int64_t shift_feature_frames_next = 32;
    int64_t pre_encode_cache_frames_first = 0;
    int64_t pre_encode_cache_frames_next = 9;
    int64_t drop_extra_pre_encoded_first = 2;
    int64_t drop_extra_pre_encoded_next = 2;
    int64_t valid_out_len = 4;
    int64_t right_context = 3;
    int64_t prompt_id = 0;
    bool strip_lang_tags = true;
    std::string onnx_dtype = "float16";
    std::unordered_map<std::string, int64_t> prompt_dictionary;
    std::vector<int64_t> att_context_size;
    std::vector<int64_t> attention_cache_shape;
    std::vector<int64_t> convolution_cache_shape;
    std::vector<float> decoder_init_hidden;
    std::vector<float> decoder_init_cell;

    [[nodiscard]] double FrameRate() const
    {
        return static_cast<double>(hop_length) / static_cast<double>(sampling_rate) *
               static_cast<double>(subsampling_factor);
    }

    static Metadata Load(const std::filesystem::path& path);
    void ApplyLangIdOverride(const std::string& lang_id);
};

class Tokenizer : public din::io::Tokenizer
{
public:
    explicit Tokenizer(const std::string& path)
        : din::io::Tokenizer(path, din::io::TokenizerFormat::Vocab)
    {
    }

    [[nodiscard]] std::string Decode(const std::vector<int64_t>& ids, bool strip_lang_tags = true) const
    {
        return din::io::Tokenizer::Decode(ids, false, strip_lang_tags);
    }
};

struct GenerateOutput
{
    std::vector<int64_t> tokens;
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

struct NemotronConfig
{
    std::filesystem::path model_dir = "artifacts/nemotron/onnx";
    std::string provider = "cpu";
    std::filesystem::path ep_cache_dir = "artifacts/nemotron/trt_rtx_cache";
    std::filesystem::path ep_context_dir = "artifacts/nemotron/ep_context";
    std::string lang_id = "auto";
};

class NemotronPipeline
{
public:
    explicit NemotronPipeline(NemotronConfig config);

    [[nodiscard]] TranscriptionResult Transcribe(const Audio& audio);
    [[nodiscard]] TranscriptionResult TranscribeFile(const std::filesystem::path& audio_path);
    void Print(std::ostream& stream, const TranscriptionResult& result, const TranscriptionOptions& options) const;

private:
    std::unique_ptr<din::common::OrtRunner> MakeRunner(const std::filesystem::path& path,
                                                       const din::common::EpContextOptions& ep_context,
                                                       const din::common::ModelProfile& profile = {});
    std::vector<Ort::Value> RunPreprocessor(const Audio& audio);
    void RunStreamingTranscription(Ort::Value& input_features, Ort::Value& feature_lengths, GenerateOutput& output);

    NemotronConfig config_;
    Metadata metadata_;
    std::unique_ptr<Tokenizer> tokenizer_;
    Ort::Env env_;
    Ort::SyncStream compute_stream_{nullptr};
    std::unique_ptr<din::common::OrtRunner> preprocessor_;
    std::unique_ptr<din::common::OrtRunner> encoder_;
    std::unique_ptr<din::common::OrtRunner> predict_;
};

}  // namespace din::asr::nemotron
