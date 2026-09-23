// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <string>

namespace din::common
{
enum class ProgressStage
{
    DecodingAudio,
    LoadingModel,
    CompilingModel,
    Transcribing,
    Aligning
};
struct InferenceProgress
{
    ProgressStage stage;
    std::string detail;
    double completed_audio_seconds = 0;
    double total_audio_seconds = 0;
};
// Invoked synchronously on the pipeline's calling thread. Keep callbacks short.
// Optional: existing CLI callers need not install a callback.
using ProgressCallback = std::function<void(const InferenceProgress&)>;
}  // namespace din::common
