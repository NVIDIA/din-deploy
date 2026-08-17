// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "asr_bindings.h"
#include "parakeet_tdt.h"

namespace din::asr::parakeet
{

namespace nb = nanobind;

nb::dict ToDict(const TranscriptionResult& result)
{
    nb::dict output;
    output["text"] = result.text;
    output["tokens"] = result.generated.tokens;
    output["durations"] = result.generated.durations;
    output["starts"] = result.generated.starts;
    return output;
}

}  // namespace din::asr::parakeet

NB_MODULE(_parakeet_tdt_cpp, module)
{
    din::asr::common::BindTranscriber<din::asr::parakeet::ParakeetPipeline, din::asr::parakeet::ParakeetConfig>(
        module, "nanobind wrapper for the DIN Deploy Parakeet TDT C++ ONNX pipeline",
        {.model_dir = "artifacts/parakeet/onnx",
         .provider = "trt-rtx",
         .ep_cache = "artifacts/parakeet/trt_rtx_cache",
         .ep_context = "artifacts/parakeet/ep_context"});
}
