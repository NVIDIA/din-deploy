// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "asr_bindings.h"
#include "nemotron.h"

namespace din::asr::nemotron
{

namespace nb = nanobind;

nb::dict ToDict(const TranscriptionResult& result)
{
    nb::dict output;
    output["text"] = result.text;
    output["tokens"] = result.generated.tokens;
    output["starts"] = result.generated.starts;
    return output;
}

}  // namespace din::asr::nemotron

NB_MODULE(_nemotron_asr_cpp, module)
{
    din::asr::common::BindTranscriber<din::asr::nemotron::NemotronPipeline, din::asr::nemotron::NemotronConfig>(
        module, "nanobind wrapper for the DIN Deploy Nemotron C++ ONNX pipeline",
        {.model_dir = "artifacts/nemotron/onnx",
         .provider = "cpu",
         .ep_cache = "artifacts/nemotron/trt_rtx_cache",
         .ep_context = "artifacts/nemotron/ep_context"});
}
