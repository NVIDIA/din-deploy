// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "asr_common.h"
#include "nemotron.h"

int main(int argc, char** argv)
{
    const din::asr::common::CliSpec spec{
        .fallback_name = "din_asr_nemotron",
        .artifact_label = "Nemotron",
        .default_provider = "cpu",
        .timestamps_modes = "none|token",
        .allowed_timestamps = {"none", "token"},
        .allowed_lang_ids =
            {
                "auto",  "en-US", "en",    "en-GB", "enGB",   "es-ES", "esES",   "es-US", "es",    "zh-CN", "zh-ZH",
                "zh-TW", "hi-IN", "hi",    "hi-HI", "ar-AR",  "ar",    "fr-FR",  "fr",    "de-DE", "de",    "ja-JP",
                "ja-JA", "ru-RU", "ru",    "pt-BR", "pt-PT",  "pt",    "ko-KR",  "ko",    "ko-KO", "it-IT", "it",
                "nl-NL", "nl",    "pl-PL", "pl",    "tr-TR",  "tr",    "uk-UA",  "uk",    "ro-RO", "ro",    "el-GR",
                "el",    "cs-CZ", "cs",    "hu-HU", "hu",     "sv-SE", "sv",     "da-DK", "da",    "fi-FI", "fi",
                "no-NO", "no",    "nb-NO", "nb",    "nn-NO",  "nn",    "sk-SK",  "sk",    "hr-HR", "hr",    "bg-BG",
                "bg",    "lt-LT", "lt",    "et-EE", "et",     "lv-LV", "lv",     "sl-SI", "sl",    "th-TH", "vi-VN",
                "id-ID", "ms-MY", "bn-IN", "ur-PK", "fa-IR",  "ta-IN", "te-IN",  "mr-IN", "gu-IN", "kn-IN", "ml-IN",
                "si-LK", "ne-NP", "km-KH", "sw-KE", "am-ET",  "ha-NG", "zu-ZA",  "yo-NG", "ig-NG", "af-ZA", "rw-RW",
                "so-SO", "ny-MW", "ln-CD", "or-KE", "he-IL",  "ku-TR", "az-AZ",  "ka-GE", "hy-AM", "uz-UZ", "tg-TJ",
                "ky-KG", "qu-PE", "ay-BO", "gn-PY", "nah-MX", "mi-NZ", "haw-US", "sm-WS", "to-TO", "fr-CA", "mt-MT",
            },
    };
    return din::asr::common::RunCli<din::asr::nemotron::NemotronPipeline, din::asr::nemotron::NemotronConfig>(
        argc, argv, spec);
}
