// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "audio.h"

namespace din::asr::qwen3
{
enum class AlignmentGranularity
{
    Words,
    Characters
};
struct ForcedAlignerConfig
{
    std::string provider = "trt-rtx";
    std::filesystem::path model_dir = "artifacts/qwen3/aligner-onnx-bf16";
    std::filesystem::path ep_cache_dir = "artifacts/qwen3/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/qwen3/ep_context";
    AlignmentGranularity granularity = AlignmentGranularity::Words;
};

struct WordTimestamp
{
    std::string text;
    float start_time = 0;
    float end_time = 0;
};

struct AlignmentSegment
{
    std::string text;
    // Half-open offsets in the supplied mono 16 kHz audio.
    size_t start_sample = 0;
    size_t end_sample = 0;
    std::string language = "English";
};

// Accepts text from any recognizer; no ASR model is loaded. One synchronous call per instance.
class Qwen3ForcedAligner
{
public:
    explicit Qwen3ForcedAligner(ForcedAlignerConfig config = {});
    ~Qwen3ForcedAligner();
    std::vector<WordTimestamp> Align(const din::io::Audio& audio, const std::string& text,
                                     const std::string& language = "English");
    std::vector<WordTimestamp> AlignFile(const std::filesystem::path& path, const std::string& text,
                                         const std::string& language = "English");
    std::vector<WordTimestamp> AlignUnits(const din::io::Audio& audio, const std::vector<std::string>& units);
    // Returns recording-relative timestamps in segment order; overlaps are preserved.
    std::vector<WordTimestamp> AlignSegments(const din::io::Audio& audio, std::span<const AlignmentSegment> segments);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace din::asr::qwen3
