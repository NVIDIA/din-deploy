// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio.h"
#include "progress.h"

namespace din::asr::qwen3
{
struct ForcedAlignerConfig
{
    std::string provider = "trt-rtx";
    std::filesystem::path model_dir = "artifacts/qwen3/aligner-onnx-bf16";
    std::filesystem::path ep_cache_dir = "artifacts/qwen3/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/qwen3/ep_context";
    din::common::ProgressCallback progress;
};

struct WordTimestamp
{
    std::string text;
    float start_time = 0;
    float end_time = 0;
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace din::asr::qwen3
