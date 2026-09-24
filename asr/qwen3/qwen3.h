// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "audio.h"

namespace din::asr::qwen3
{
struct Qwen3Config
{
    std::string provider = "trt-rtx";
    std::filesystem::path model_dir = "artifacts/qwen3/onnx-bf16";
    std::filesystem::path ep_cache_dir = "artifacts/qwen3/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/qwen3/ep_context";
    std::string lang_id = "auto";
    int max_new_tokens = 1024;
    int max_chunk_seconds = 0;  // Auto: 1200 s, limited by KV capacity; boundaries add up to 5 s.
};

struct TranscriptionSegment
{
    std::string text;
    std::string language;
    // Half-open sample offsets in normalized 16 kHz audio, not word timestamps.
    size_t start_sample = 0;
    size_t end_sample = 0;
};

struct TranscriptionResult
{
    std::string text;
    std::string language;
    std::vector<int64_t> tokens;
    std::vector<TranscriptionSegment> segments;
    bool reached_eos = false;
    size_t chunks_processed = 0;
    float audio_seconds = 0;
    float transcribe_seconds = 0;
};

struct StreamingConfig
{
    float chunk_seconds = 2.f;
    int unfixed_chunks = 2;
    int unfixed_tokens = 5;
};

struct StreamingResult
{
    std::string text;
    std::string language;
    size_t samples_processed = 0;
    size_t updates = 0;
    bool reached_eos = true;
    bool final = false;
};

// One synchronous stream per instance. Reuse the instance across files.
class Qwen3Pipeline
{
public:
    explicit Qwen3Pipeline(Qwen3Config config);
    ~Qwen3Pipeline();
    Qwen3Pipeline(const Qwen3Pipeline&) = delete;
    Qwen3Pipeline& operator=(const Qwen3Pipeline&) = delete;
    TranscriptionResult Transcribe(const din::io::Audio& audio);
    TranscriptionResult TranscribeFile(const std::filesystem::path& path);
    // One utterance per stream. Each update replaces the previous hypothesis.
    void StartStream(StreamingConfig config = {});
    std::vector<StreamingResult> PushAudio(std::span<const float> pcm16k);
    StreamingResult FinishStream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace din::asr::qwen3
