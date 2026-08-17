// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "asr_common.h"
#include "parakeet_tdt.h"

int main(int argc, char** argv)
{
    const din::asr::common::CliSpec spec{
        .fallback_name = "din_asr_parakeet_tdt",
        .artifact_label = "Parakeet",
        .default_provider = "trt-rtx",
        .timestamps_modes = "none|segment|token",
        .allowed_timestamps = {"none", "segment", "token"},
    };
    return din::asr::common::RunCli<din::asr::parakeet::ParakeetPipeline, din::asr::parakeet::ParakeetConfig>(
        argc, argv, spec);
}
