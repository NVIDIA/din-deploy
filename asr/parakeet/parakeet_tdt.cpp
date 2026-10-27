// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "parakeet_tdt.h"

#include <nvtx_helper.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <ort_session.h>

namespace din::asr::parakeet
{

namespace fs = std::filesystem;

using din::asr::common::SecondsSince;
using din::common::MakeHostTensor;
using din::common::OrtRunner;
using din::common::TensorBuffer;

Metadata Metadata::Load(const fs::path& path)
{
    const auto data = nlohmann::json::parse(din::asr::common::ReadText(path.string()));
    Metadata metadata;
    metadata.blank_token_id = data.at("blank_token_id").get<int64_t>();
    metadata.pad_token_id = data.at("pad_token_id").get<int64_t>();
    metadata.vocab_size = data.at("vocab_size").get<int64_t>();
    metadata.max_symbols_per_step = data.at("max_symbols_per_step").get<int64_t>();
    metadata.decoder_hidden_size = data.at("decoder_hidden_size").get<int64_t>();
    metadata.joint_dim = data.value("joint_dim", metadata.decoder_hidden_size);
    metadata.num_decoder_layers = data.at("num_decoder_layers").get<int64_t>();
    metadata.sampling_rate = data.at("sampling_rate").get<int64_t>();
    metadata.hop_length = data.at("hop_length").get<int64_t>();
    metadata.subsampling_factor = data.at("subsampling_factor").get<int64_t>();
    metadata.encoder_profile_frames = data.value("encoder_profile_frames", metadata.encoder_profile_frames);
    metadata.durations = data.at("durations").get<std::vector<int64_t>>();
    return metadata;
}

namespace
{

Ort::Value RunEncoder(OrtRunner& encoder, Ort::Value& input_features, Ort::Value& attention_mask, bool use_device_io)
{
    din::common::nvtx_scoped_range range{"encoder"};
    TensorBuffer<float> features(encoder, input_features.GetTensorTypeAndShapeInfo().GetShape(), use_device_io);
    TensorBuffer<bool> mask(encoder, attention_mask.GetTensorTypeAndShapeInfo().GetShape(), use_device_io);
    features.CopyFromHostValue(input_features);
    mask.CopyFromHostValue(attention_mask);
    features.CopyAsyncToDevice();
    mask.CopyAsyncToDevice();

    Ort::IoBinding binding(encoder.session);
    binding.BindInput("input_features", features.BindingValue());
    binding.BindInput("attention_mask", mask.BindingValue());
    din::common::BindOutput(binding, "encoder_states", encoder, use_device_io);
    din::common::BindOutput(binding, "encoder_mask", encoder, use_device_io);

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    encoder.session.Run(run_options, binding);

    auto outputs = binding.GetOutputValues();
    if (outputs.empty())
    {
        throw std::runtime_error("encoder run returned no outputs");
    }
    return std::move(outputs[0]);
}

Ort::Value FrameView(OrtRunner& runner, Ort::Value& encoder_states, int64_t frame, int64_t joint_dim,
                     bool use_device_io)
{
    const std::vector<int64_t> frame_shape{1, 1, joint_dim};
    auto* data = encoder_states.GetTensorMutableData<float>() + static_cast<size_t>(frame * joint_dim);
    if (use_device_io)
    {
        return Ort::Value::CreateTensor<float>(runner.DeviceMemory(), data, static_cast<size_t>(joint_dim),
                                               frame_shape.data(), frame_shape.size());
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory, data, static_cast<size_t>(joint_dim), frame_shape.data(),
                                           frame_shape.size());
}

GenerateOutput GreedyDecode(OrtRunner& predict, const Metadata& metadata, Ort::Value& encoder_states,
                            bool use_device_io)
{
    din::common::nvtx_scoped_range range{"greedy_decode"};
    const auto encoded_frames = encoder_states.GetTensorTypeAndShapeInfo().GetShape().at(1);
    const std::vector<int64_t> id_shape{1, 1};
    const std::vector<int64_t> control_shape{1, 1, 2};
    const std::vector<int64_t> frame_shape{1, 1, metadata.joint_dim};
    const std::vector<int64_t> state_shape{metadata.num_decoder_layers, 1, metadata.decoder_hidden_size};

    TensorBuffer<int64_t> input_id(predict, id_shape, use_device_io);
    TensorBuffer<int64_t> control(predict, control_shape, use_device_io);
    TensorBuffer<float> hidden(predict, state_shape, use_device_io);
    TensorBuffer<float> cell(predict, state_shape, use_device_io);
    TensorBuffer<float> next_hidden(predict, state_shape, use_device_io);
    TensorBuffer<float> next_cell(predict, state_shape, use_device_io);

    *input_id.HostData() = metadata.blank_token_id;
    hidden.Fill(0.0F);
    cell.Fill(0.0F);
    input_id.CopyAsyncToDevice();
    hidden.CopyAsyncToDevice();
    cell.CopyAsyncToDevice();

    Ort::Value encoder_frame;
    std::vector<float> host_frame_storage;
    if (use_device_io)
    {
        encoder_frame =
            Ort::Value::CreateTensor<float>(predict.DeviceAllocator(), frame_shape.data(), frame_shape.size());
    }
    else
    {
        host_frame_storage.resize(static_cast<size_t>(metadata.joint_dim));
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        encoder_frame = Ort::Value::CreateTensor<float>(memory, host_frame_storage.data(), host_frame_storage.size(),
                                                        frame_shape.data(), frame_shape.size());
    }

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

    Ort::IoBinding binding(predict.session);
    binding.BindInput("input_id", input_id.BindingValue());
    binding.BindInput("hidden", hidden.BindingValue());
    binding.BindInput("cell", cell.BindingValue());
    binding.BindInput("encoder_frame", encoder_frame);
    binding.BindOutput("control", control.BindingValue());
    binding.BindOutput("next_hidden", next_hidden.BindingValue());
    binding.BindOutput("next_cell", next_cell.BindingValue());

    GenerateOutput output;
    output.tokens.reserve(static_cast<size_t>(encoded_frames));
    output.durations.reserve(static_cast<size_t>(encoded_frames));
    output.starts.reserve(static_cast<size_t>(encoded_frames));

    int64_t time_idx = 0;
    while (time_idx < encoded_frames)
    {
        int64_t symbols_added = 0;
        int64_t skip = 0;

        while (symbols_added < metadata.max_symbols_per_step)
        {
            auto frame_view = FrameView(predict, encoder_states, time_idx, metadata.joint_dim, use_device_io);
            if (use_device_io)
            {
                Ort::ThrowOnError(predict.env.CopyTensor(frame_view, encoder_frame, *predict.compute_stream));
            }
            else
            {
                std::copy_n(frame_view.GetTensorData<float>(), static_cast<size_t>(metadata.joint_dim),
                            encoder_frame.GetTensorMutableData<float>());
            }

            {
                din::common::nvtx_scoped_range predict_range{"predict_step"};
                predict.session.Run(run_options, binding);
            }

            control.CopyAsyncToHostWithNotification().Sync();
            const auto* control_data = control.HostData();
            const auto token = control_data[0];
            const auto duration_index = control_data[1];
            skip = metadata.durations[static_cast<size_t>(duration_index)];

            if (token != metadata.blank_token_id)
            {
                output.tokens.push_back(token);
                output.durations.push_back(skip);
                output.starts.push_back(time_idx);

                hidden.CopyFrom(next_hidden);
                cell.CopyFrom(next_cell);
                *input_id.HostData() = token;
                input_id.CopyAsyncToDevice();
            }

            symbols_added += 1;
            time_idx += skip;
            if (skip != 0)
            {
                break;
            }
        }

        if (skip == 0 || symbols_added == metadata.max_symbols_per_step)
        {
            time_idx += 1;
        }
    }

    return output;
}

}  // namespace

ParakeetPipeline::ParakeetPipeline(ParakeetConfig config)
    : config_(std::move(config))
    , env_(ORT_LOGGING_LEVEL_WARNING, "din_asr_parakeet_tdt")
{
    DIN_NVTX_FUNC_RANGE();
    const fs::path model_dir = config_.model_dir;
    metadata_ = Metadata::Load(din::asr::common::RequirePath(model_dir / "metadata.json", "metadata.json"));
    if (config_.encoder_profile_frames == ParakeetConfig{}.encoder_profile_frames)
    {
        config_.encoder_profile_frames = metadata_.encoder_profile_frames;
    }
    tokenizer_ =
        std::make_unique<Tokenizer>(din::asr::common::RequirePath(model_dir / "tokenizer.json", "tokenizer.json"));

    if (config_.provider == "trt-rtx" || config_.provider == "trt")
    {
        din::common::RegisterTensorRTRTXProvider(env_);
        compute_stream_ = din::common::CreateTensorRTRTXComputeStream(env_);
    }

    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config_.ep_context_dir.string();

    din::common::ModelProfile preprocessor_profile;
    {
        const auto max_s = std::max<int64_t>(2560, config_.encoder_profile_frames * 160);
        const auto opt_s = std::min<int64_t>(655360, max_s);
        preprocessor_profile.min_shapes = "input_signal:1x2560,lengths:1";
        preprocessor_profile.opt_shapes = "input_signal:1x" + std::to_string(opt_s) + ",lengths:1";
        preprocessor_profile.max_shapes = "input_signal:1x" + std::to_string(max_s) + ",lengths:1";
        preprocessor_profile.cache_subpath = "preprocessor_s" + std::to_string(max_s);
    }

    din::common::ModelProfile encoder_profile;
    {
        const auto max_f = std::max<int64_t>(16, config_.encoder_profile_frames);
        const auto opt_f = std::min<int64_t>(4096, max_f);
        encoder_profile.min_shapes = "input_features:1x16x128,attention_mask:1x16";
        encoder_profile.opt_shapes =
            "input_features:1x" + std::to_string(opt_f) + "x128,attention_mask:1x" + std::to_string(opt_f);
        encoder_profile.max_shapes =
            "input_features:1x" + std::to_string(max_f) + "x128,attention_mask:1x" + std::to_string(max_f);
        encoder_profile.cache_subpath = "encoder_f" + std::to_string(max_f);
    }

    din::common::ModelProfile predict_profile;
    {
        constexpr auto shapes = "input_id:1x1,hidden:2x1x640,cell:2x1x640,encoder_frame:1x1x640";
        predict_profile.min_shapes = shapes;
        predict_profile.opt_shapes = shapes;
        predict_profile.max_shapes = shapes;
        predict_profile.cache_subpath = "predict_step";
    }

    preprocessor_ = MakeRunner(model_dir / "preprocessor.onnx", ep_context, preprocessor_profile);
    encoder_ = MakeRunner(model_dir / "encoder.onnx", ep_context, encoder_profile);
    predict_ = MakeRunner(model_dir / "predict_step.onnx", ep_context, predict_profile);
}

std::unique_ptr<OrtRunner> ParakeetPipeline::MakeRunner(const fs::path& path,
                                                        const din::common::EpContextOptions& ep_context,
                                                        const din::common::ModelProfile& profile)
{
    const auto model_path = din::asr::common::RequirePath(path, path.filename().string());
    auto* stream = (config_.provider == "trt-rtx" || config_.provider == "trt") ? &compute_stream_ : nullptr;
    return std::make_unique<OrtRunner>(env_, model_path, config_.provider, config_.ep_cache_dir.string(), ep_context,
                                       profile, stream);
}

TranscriptionResult ParakeetPipeline::Transcribe(const Audio& audio)
{
    DIN_NVTX_FUNC_RANGE();
    if (audio.sample_rate != metadata_.sampling_rate)
    {
        throw std::runtime_error("expected audio input to match model sample rate");
    }

    const auto transcribe_start = std::chrono::steady_clock::now();
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> input_signal = audio.samples;
    std::vector<int64_t> lengths{static_cast<int64_t>(input_signal.size())};

    std::vector<Ort::Value> preprocessor_inputs;
    preprocessor_inputs.emplace_back(
        MakeHostTensor(memory, input_signal, {1, static_cast<int64_t>(input_signal.size())}));
    preprocessor_inputs.emplace_back(MakeHostTensor(memory, lengths, {1}));

    Ort::RunOptions run_options;
    std::vector<Ort::Value> preprocessor_outputs;
    {
        din::common::nvtx_scoped_range preprocessor_range{"preprocessor"};
        const std::vector<const char*> preprocessor_input_names = {"input_signal", "lengths"};
        const std::vector<const char*> preprocessor_output_names = {"input_features", "attention_mask"};
        preprocessor_outputs = preprocessor_->session.Run(
            run_options, preprocessor_input_names.data(), preprocessor_inputs.data(), preprocessor_inputs.size(),
            preprocessor_output_names.data(), preprocessor_output_names.size());
    }

    TranscriptionResult result;
    result.audio_seconds = audio.Duration();

    const bool use_device_io = predict_->HasDeviceIo() && encoder_->HasDeviceIo();
    Ort::Value encoder_states{nullptr};

    const auto encode_start = std::chrono::steady_clock::now();
    encoder_states = RunEncoder(*encoder_, preprocessor_outputs[0], preprocessor_outputs[1], use_device_io);
    result.encode_seconds = SecondsSince(encode_start);

    const auto greedy_start = std::chrono::steady_clock::now();
    result.generated = GreedyDecode(*predict_, metadata_, encoder_states, use_device_io);
    result.greedy_seconds = SecondsSince(greedy_start);

    result.transcribe_seconds = SecondsSince(transcribe_start);
    result.text = tokenizer_->Decode(result.generated.tokens);
    return result;
}

TranscriptionResult ParakeetPipeline::TranscribeFile(const fs::path& audio_path)
{
    const auto audio = din::io::LoadAudio(audio_path.string(), static_cast<int>(metadata_.sampling_rate));
    return Transcribe(audio);
}

void ParakeetPipeline::Print(std::ostream& stream, const TranscriptionResult& result,
                             const TranscriptionOptions& options) const
{
    DIN_NVTX_FUNC_RANGE();
    if (options.timestamps == "none")
    {
        stream << result.text << '\n';
        return;
    }

    const auto frame_rate = metadata_.FrameRate();
    if (options.timestamps == "token")
    {
        for (size_t i = 0; i < result.generated.tokens.size(); ++i)
        {
            const auto start = result.generated.starts[i] * frame_rate;
            const auto end = (result.generated.starts[i] + result.generated.durations[i]) * frame_rate;
            stream << start << "s - " << end << "s : " << tokenizer_->CleanToken(result.generated.tokens[i]) << " ["
                   << result.generated.tokens[i] << "]\n";
        }
        return;
    }

    if (options.timestamps == "segment")
    {
        const auto start = result.generated.starts.empty() ? 0.0 : result.generated.starts.front() * frame_rate;
        const auto end = result.generated.starts.empty()
                             ? 0.0
                             : (result.generated.starts.back() + result.generated.durations.back()) * frame_rate;
        stream << start << "s - " << end << "s : " << result.text << '\n';
    }
}

}  // namespace din::asr::parakeet
