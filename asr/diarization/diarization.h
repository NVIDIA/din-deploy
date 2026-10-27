// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio.h"

namespace din::asr::diarization
{
struct Config
{
    std::string provider = "trt-rtx";
    std::filesystem::path model_dir = "artifacts/nemotron-diarization/onnx-bf16";
    std::filesystem::path ep_cache_dir = "artifacts/nemotron-diarization/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/nemotron-diarization/ep_context";
    float threshold = 0.5f;
};

struct Segment
{
    float start_time = 0;
    float end_time = 0;
    int speaker = -1;
};

struct Result
{
    std::vector<Segment> segments;
    float audio_seconds = 0;
    float process_seconds = 0;
    // Greatest cumulative overlap; -1 means no active speaker.
    int SpeakerAt(float start, float end) const;
};

// One synchronous stream per instance. Speaker state resets between recordings.
class Pipeline
{
public:
    explicit Pipeline(Config config = {});
    ~Pipeline();
    Result Diarize(const din::io::Audio& audio);
    Result DiarizeFile(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace din::asr::diarization
