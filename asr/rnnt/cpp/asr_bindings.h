// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "asr_common.h"
#include "audio.h"
#include "ort_helper.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace din::asr::common
{

namespace nb = nanobind;

struct BindingDefaults
{
    const char* model_dir;
    const char* provider;
    const char* ep_cache;
    const char* ep_context;
};

// Generic nanobind transcriber. The per-model result is converted to a dict
// through an unqualified ToDict() resolved by argument-dependent lookup in the
// model's namespace, which is the only model-specific piece of the binding.
template <class Pipeline, class Config>
class Transcriber
{
public:
    Transcriber(const std::string& model_dir, const std::string& provider, const std::string& ep_cache,
                const std::string& ep_context_dir, const std::string& lang_id)
    {
        Config config;
        config.model_dir = model_dir;
        config.provider = provider;
        config.ep_cache_dir = ep_cache;
        config.ep_context_dir = ep_context_dir;
        if constexpr (requires(Config value) { value.lang_id = std::string{}; })
        {
            config.lang_id = lang_id;
        }
        pipeline_ = std::make_unique<Pipeline>(std::move(config));
    }

    nb::dict Transcribe(nb::ndarray<nb::numpy, const float, nb::shape<-1>, nb::c_contig> samples, int sample_rate)
    {
        din::io::Audio audio;
        audio.sample_rate = sample_rate;
        audio.samples.assign(samples.data(), samples.data() + samples.shape(0));

        nb::gil_scoped_release release;
        const auto result = pipeline_->Transcribe(audio);
        nb::gil_scoped_acquire acquire;
        return ToDict(result);
    }

    nb::dict TranscribeFile(const std::string& path)
    {
        nb::gil_scoped_release release;
        const auto result = pipeline_->TranscribeFile(path);
        nb::gil_scoped_acquire acquire;
        return ToDict(result);
    }

private:
    std::unique_ptr<Pipeline> pipeline_;
};

template <class Pipeline, class Config>
void BindTranscriber(nb::module_& module, const char* doc, const BindingDefaults& defaults)
{
    using Bound = Transcriber<Pipeline, Config>;
    module.doc() = doc;
    nb::class_<Bound>(module, "Transcriber")
        .def(nb::init<const std::string&, const std::string&, const std::string&, const std::string&,
                      const std::string&>(),
             nb::arg("model_dir") = defaults.model_dir, nb::arg("provider") = defaults.provider,
             nb::arg("ep_cache") = defaults.ep_cache, nb::arg("ep_context_dir") = defaults.ep_context,
             nb::arg("lang_id") = "auto")
        .def("transcribe", &Bound::Transcribe, nb::arg("samples"), nb::arg("sample_rate"))
        .def("transcribe_file", &Bound::TranscribeFile, nb::arg("path"));
}

}  // namespace din::asr::common
