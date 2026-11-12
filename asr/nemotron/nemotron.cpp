// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nemotron.h"

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
#include <vector>

#include "nvtx_helper.h"
#include <nlohmann/json.hpp>
#include <onnxruntime_run_options_config_keys.h>

namespace din::asr::nemotron
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
    metadata.max_symbols_per_step = data.at("max_symbols_per_step").get<int64_t>();
    metadata.decoder_hidden_size = data.at("decoder_hidden_size").get<int64_t>();
    metadata.joint_dim = data.at("joint_dim").get<int64_t>();
    metadata.num_decoder_layers = data.at("num_decoder_layers").get<int64_t>();
    metadata.sampling_rate = data.at("sampling_rate").get<int64_t>();
    metadata.hop_length = data.at("hop_length").get<int64_t>();
    metadata.num_mel_bins = data.value("num_mel_bins", metadata.num_mel_bins);
    metadata.subsampling_factor = data.at("subsampling_factor").get<int64_t>();
    metadata.preprocessor_profile_frames =
        data.value("preprocessor_profile_frames", metadata.preprocessor_profile_frames);
    metadata.chunk_size = data.value("chunk_size", metadata.chunk_size);
    metadata.encoder_step_input_frames = data.value("encoder_step_input_frames", metadata.encoder_step_input_frames);
    metadata.max_mel_frames = data.value("max_mel_frames", metadata.encoder_step_input_frames);
    metadata.encoder_step_input_frames = metadata.max_mel_frames;
    metadata.chunk_feature_frames_first = data.value("chunk_feature_frames_first", metadata.chunk_feature_frames_first);
    metadata.chunk_feature_frames_next = data.value("chunk_feature_frames_next", metadata.chunk_feature_frames_next);
    metadata.shift_feature_frames_first = data.value("shift_feature_frames_first", metadata.shift_feature_frames_first);
    metadata.shift_feature_frames_next = data.value("shift_feature_frames_next", metadata.shift_feature_frames_next);
    metadata.pre_encode_cache_frames_first =
        data.value("pre_encode_cache_frames_first", metadata.pre_encode_cache_frames_first);
    metadata.pre_encode_cache_frames_next =
        data.value("pre_encode_cache_frames_next", metadata.pre_encode_cache_frames_next);
    metadata.drop_extra_pre_encoded_first =
        data.value("drop_extra_pre_encoded_first", metadata.drop_extra_pre_encoded_first);
    metadata.drop_extra_pre_encoded_next =
        data.value("drop_extra_pre_encoded_next", metadata.drop_extra_pre_encoded_next);
    metadata.valid_out_len = data.value("valid_out_len", metadata.valid_out_len);
    metadata.right_context = data.value("right_context", metadata.right_context);
    metadata.prompt_id = data.at("prompt_id").get<int64_t>();
    metadata.onnx_dtype = data.value("onnx_dtype", metadata.onnx_dtype);
    metadata.prompt_dictionary = data.value("prompt_dictionary", std::unordered_map<std::string, int64_t>{});
    metadata.strip_lang_tags = data.value("strip_lang_tags", metadata.strip_lang_tags);
    metadata.att_context_size = data.value("att_context_size", metadata.att_context_size);
    metadata.attention_cache_shape = data.value("attention_cache_shape", metadata.attention_cache_shape);
    metadata.convolution_cache_shape = data.value("convolution_cache_shape", metadata.convolution_cache_shape);
    metadata.decoder_init_hidden = din::asr::common::ReadFloatFile(path.parent_path() / "decoder_init_hidden.f32");
    metadata.decoder_init_cell = din::asr::common::ReadFloatFile(path.parent_path() / "decoder_init_cell.f32");
    return metadata;
}

void Metadata::ApplyLangIdOverride(const std::string& lang_id)
{
    const auto selected = lang_id.empty() ? std::string{"auto"} : lang_id;
    if (prompt_dictionary.empty())
    {
        if (selected == "auto")
        {
            return;
        }
        throw std::runtime_error("metadata does not contain prompt_dictionary for lang-id: " + selected);
    }
    const auto iter = prompt_dictionary.find(selected);
    if (iter == prompt_dictionary.end())
    {
        throw std::runtime_error("unsupported lang-id: " + selected);
    }
    prompt_id = iter->second;
}

namespace
{

int64_t FillStreamingChunk(const float* feature_data, int64_t total_feature_frames, size_t feature_bins, int64_t start,
                           int64_t chunk_feature_frames, int64_t pre_encode_cache_frames, int64_t input_feature_frames,
                           float* chunk_data)
{
    const auto value_count = static_cast<size_t>(input_feature_frames) * feature_bins;
    std::fill_n(chunk_data, value_count, 0.0F);

    const auto actual_frames = std::min<int64_t>(chunk_feature_frames, total_feature_frames - start);
    const auto prefix_frames = std::min<int64_t>(pre_encode_cache_frames, start);
    const auto prefix_start = start - prefix_frames;
    const auto prefix_destination = pre_encode_cache_frames - prefix_frames;
    if (prefix_frames > 0)
    {
        std::copy_n(feature_data + static_cast<size_t>(prefix_start) * feature_bins,
                    static_cast<size_t>(prefix_frames) * feature_bins,
                    chunk_data + static_cast<size_t>(prefix_destination) * feature_bins);
    }
    if (actual_frames > 0)
    {
        std::copy_n(feature_data + static_cast<size_t>(start) * feature_bins,
                    static_cast<size_t>(actual_frames) * feature_bins,
                    chunk_data + static_cast<size_t>(pre_encode_cache_frames) * feature_bins);
    }
    return pre_encode_cache_frames + actual_frames;
}

int64_t EncodedFrameCount(const Metadata& metadata, int64_t chunk_length)
{
    const auto effective_frames = chunk_length - metadata.pre_encode_cache_frames_next;
    if (effective_frames <= 0)
    {
        return 0;
    }
    const auto encoded = (effective_frames + metadata.subsampling_factor - 1) / metadata.subsampling_factor;
    return std::min(encoded, metadata.valid_out_len);
}

void RunEncoderStep(OrtRunner& encoder, Ort::IoBinding& binding, TensorBuffer<float>& input_features,
                    TensorBuffer<int32_t>& feature_lengths)
{
    din::common::nvtx_scoped_range range{"encoder_step"};
    input_features.CopyAsyncToDevice();
    feature_lengths.CopyAsyncToDevice();

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    encoder.session.Run(run_options, binding);
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

void GreedyDecodeChunk(OrtRunner& predict, const Metadata& metadata, Ort::IoBinding& io_binding, Ort::Value& frame,
                       Ort::Value& device_encoder_states, int64_t encoded_frames, int64_t encoded_frame_offset,
                       int64_t base_frame, TensorBuffer<int64_t>& input_id, TensorBuffer<float>& hidden,
                       TensorBuffer<float>& cell, TensorBuffer<int64_t>& token_id, TensorBuffer<float>& next_hidden,
                       TensorBuffer<float>& next_cell, bool use_device_io, GenerateOutput& output)
{
    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

    for (int64_t time_idx = 0; time_idx < encoded_frames; ++time_idx)
    {
        int64_t symbols_added = 0;
        while (symbols_added < metadata.max_symbols_per_step)
        {
            auto frame_view = FrameView(predict, device_encoder_states, encoded_frame_offset + time_idx,
                                        metadata.joint_dim, use_device_io);
            if (use_device_io)
            {
                Ort::ThrowOnError(predict.env.CopyTensor(frame_view, frame, *predict.compute_stream));
            }
            else
            {
                std::copy_n(frame_view.GetTensorData<float>(), static_cast<size_t>(metadata.joint_dim),
                            frame.GetTensorMutableData<float>());
            }
            {
                din::common::nvtx_scoped_range predict_range{"predict_step"};
                predict.session.Run(run_options, io_binding);
            }

            token_id.CopyAsyncToHostWithNotification().Sync();
            const auto token = token_id.HostData()[0];
            if (token == metadata.blank_token_id)
            {
                break;
            }
            hidden.CopyFrom(next_hidden);
            cell.CopyFrom(next_cell);

            output.tokens.push_back(token);
            output.starts.push_back(base_frame + time_idx);
            symbols_added += 1;
        }
    }
}

}  // namespace

NemotronPipeline::NemotronPipeline(NemotronConfig config)
    : config_(std::move(config))
    , env_(ORT_LOGGING_LEVEL_WARNING, "din_asr_nemotron")
{
    DIN_NVTX_FUNC_RANGE();
    const fs::path model_dir = config_.model_dir;
    metadata_ = Metadata::Load(din::asr::common::RequirePath(model_dir / "metadata.json", "metadata.json"));
    metadata_.ApplyLangIdOverride(config_.lang_id);
    tokenizer_ = std::make_unique<Tokenizer>(din::asr::common::RequirePath(model_dir / "vocab.txt", "vocab.txt"));

    if (config_.provider == "trt-rtx" || config_.provider == "trt")
    {
        din::common::RegisterTensorRTRTXProvider(env_);
        compute_stream_ = din::common::CreateTensorRTRTXComputeStream(env_);
    }

    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config_.ep_context_dir.string();

    din::common::ModelProfile preprocessor_profile;
    {
        const auto max_s = std::max<int64_t>(2560, metadata_.preprocessor_profile_frames * metadata_.hop_length);
        const auto opt_s = std::min<int64_t>(655360, max_s);
        preprocessor_profile.min_shapes = "input_signal:1x2560,lengths:1";
        preprocessor_profile.opt_shapes = "input_signal:1x" + std::to_string(opt_s) + ",lengths:1";
        preprocessor_profile.max_shapes = "input_signal:1x" + std::to_string(max_s) + ",lengths:1";
        preprocessor_profile.cache_subpath = "preprocessor_s" + std::to_string(max_s) + "_i32";
    }

    din::common::ModelProfile encoder_profile;
    {
        auto shape_string = [](const std::vector<int64_t>& dims)
        {
            std::string shape;
            for (const auto dim : dims)
            {
                if (!shape.empty())
                {
                    shape += "x";
                }
                shape += std::to_string(dim);
            }
            return shape;
        };
        auto shapes_for_raw_frames = [this, &shape_string](int64_t frames)
        {
            return "input_features:1x" + std::to_string(frames) + "x" + std::to_string(metadata_.num_mel_bins) +
                   ",feature_lengths:1,prompt:1x128,att_cache:" + shape_string(metadata_.attention_cache_shape) +
                   ",conv_cache:" + shape_string(metadata_.convolution_cache_shape) +
                   ",right_context:1,cache_last_channel_len:1";
        };
        const auto frames = metadata_.encoder_step_input_frames;
        encoder_profile.min_shapes = shapes_for_raw_frames(frames);
        encoder_profile.opt_shapes = shapes_for_raw_frames(frames);
        encoder_profile.max_shapes = shapes_for_raw_frames(frames);
        encoder_profile.cache_subpath = "encoder_step_f" + std::to_string(frames) + "_chunk" +
                                        std::to_string(metadata_.chunk_size) + "_i32_" + metadata_.onnx_dtype;
    }

    din::common::ModelProfile predict_profile;
    {
        const auto shapes = "input_id:1x1,hidden:" + std::to_string(metadata_.num_decoder_layers) + "x1x" +
                            std::to_string(metadata_.decoder_hidden_size) +
                            ",cell:" + std::to_string(metadata_.num_decoder_layers) + "x1x" +
                            std::to_string(metadata_.decoder_hidden_size) + ",encoder_frame:1x1x" +
                            std::to_string(metadata_.joint_dim);
        predict_profile.min_shapes = shapes;
        predict_profile.opt_shapes = shapes;
        predict_profile.max_shapes = shapes;
        predict_profile.cache_subpath = "predict_step";
    }

    preprocessor_ = MakeRunner(model_dir / "preprocessor.onnx", ep_context, preprocessor_profile);
    encoder_ = MakeRunner(model_dir / "encoder_step.onnx", ep_context, encoder_profile);
    predict_ = MakeRunner(model_dir / "predict_step.onnx", ep_context, predict_profile);
}

std::unique_ptr<OrtRunner> NemotronPipeline::MakeRunner(const fs::path& path,
                                                        const din::common::EpContextOptions& ep_context,
                                                        const din::common::ModelProfile& profile)
{
    const auto model_path = din::asr::common::RequirePath(path, path.filename().string());
    auto* stream = (config_.provider == "trt-rtx" || config_.provider == "trt") ? &compute_stream_ : nullptr;
    return std::make_unique<OrtRunner>(env_, model_path, config_.provider, config_.ep_cache_dir.string(), ep_context,
                                       profile, stream);
}

std::vector<Ort::Value> NemotronPipeline::RunPreprocessor(const Audio& audio)
{
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_signal = audio.samples;
    std::vector<int32_t> lengths{static_cast<int32_t>(input_signal.size())};
    std::vector<Ort::Value> preprocessor_inputs;
    preprocessor_inputs.emplace_back(MakeHostTensor(memory, input_signal, {1, lengths[0]}));
    preprocessor_inputs.emplace_back(MakeHostTensor(memory, lengths, {1}));

    Ort::RunOptions run_options;
    din::common::nvtx_scoped_range range{"preprocessor"};
    return preprocessor_->session.Run(run_options, std::vector<const char*>{"input_signal", "lengths"}.data(),
                                      preprocessor_inputs.data(), preprocessor_inputs.size(),
                                      std::vector<const char*>{"input_features", "feature_lengths"}.data(), 2);
}

void NemotronPipeline::RunStreamingTranscription(Ort::Value& input_features, Ort::Value& feature_lengths,
                                                 GenerateOutput& output)
{
    const bool use_device_io = predict_->HasDeviceIo() && encoder_->HasDeviceIo();
    const auto* feature_data = input_features.GetTensorData<float>();
    const auto total_feature_frames = static_cast<int64_t>(feature_lengths.GetTensorData<int32_t>()[0]);
    const auto feature_bins = static_cast<size_t>(metadata_.num_mel_bins);
    const auto exported_input_feature_frames = metadata_.encoder_step_input_frames;
    const std::vector<int64_t> chunk_shape{1, exported_input_feature_frames, static_cast<int64_t>(feature_bins)};
    const std::vector<int64_t> encoder_states_shape{1, metadata_.valid_out_len, metadata_.joint_dim};
    const auto state_size = static_cast<size_t>(metadata_.num_decoder_layers * metadata_.decoder_hidden_size);
    if (metadata_.decoder_init_hidden.size() != state_size || metadata_.decoder_init_cell.size() != state_size)
    {
        throw std::runtime_error("decoder initial state files are missing or have the wrong shape");
    }

    std::vector<float> prompt(128, 0.0F);
    prompt[static_cast<size_t>(metadata_.prompt_id)] = 1.0F;

    const std::vector<int64_t> id_shape{1, 1};
    const std::vector<int64_t> state_shape{metadata_.num_decoder_layers, 1, metadata_.decoder_hidden_size};
    TensorBuffer<float> att_cache(*encoder_, metadata_.attention_cache_shape, use_device_io);
    TensorBuffer<float> conv_cache(*encoder_, metadata_.convolution_cache_shape, use_device_io);
    TensorBuffer<float> next_att_cache(*encoder_, metadata_.attention_cache_shape, use_device_io);
    TensorBuffer<float> next_conv_cache(*encoder_, metadata_.convolution_cache_shape, use_device_io);
    TensorBuffer<int32_t> output_lengths(*encoder_, {1}, use_device_io);
    TensorBuffer<int32_t> next_cache_last_channel_len(*encoder_, {1}, use_device_io);
    TensorBuffer<float> encoder_states(*encoder_, encoder_states_shape, use_device_io);
    TensorBuffer<float> chunk(*encoder_, chunk_shape, use_device_io);
    TensorBuffer<int32_t> chunk_length(*encoder_, {1}, use_device_io);
    TensorBuffer<float> prompt_buffer(*encoder_, {1, 128}, use_device_io);
    TensorBuffer<int32_t> right_context(*encoder_, {1}, use_device_io);
    TensorBuffer<int32_t> cache_last_channel_len(*encoder_, {1}, use_device_io);

    TensorBuffer<int64_t> input_id(*predict_, id_shape, use_device_io);
    TensorBuffer<int64_t> token_id(*predict_, id_shape, use_device_io);
    TensorBuffer<float> hidden(*predict_, state_shape, use_device_io);
    TensorBuffer<float> cell(*predict_, state_shape, use_device_io);
    TensorBuffer<float> next_hidden(*predict_, state_shape, use_device_io);
    TensorBuffer<float> next_cell(*predict_, state_shape, use_device_io);

    att_cache.Fill(0.0F);
    conv_cache.Fill(0.0F);
    att_cache.CopyAsyncToDevice();
    conv_cache.CopyAsyncToDevice();
    std::copy(prompt.begin(), prompt.end(), prompt_buffer.HostData());
    *right_context.HostData() = static_cast<int32_t>(metadata_.right_context);
    *cache_last_channel_len.HostData() = 0;
    *input_id.HostData() = metadata_.blank_token_id;
    std::copy(metadata_.decoder_init_hidden.begin(), metadata_.decoder_init_hidden.end(), hidden.HostData());
    std::copy(metadata_.decoder_init_cell.begin(), metadata_.decoder_init_cell.end(), cell.HostData());
    input_id.CopyAsyncToDevice();
    hidden.CopyAsyncToDevice();
    cell.CopyAsyncToDevice();
    prompt_buffer.CopyAsyncToDevice();
    right_context.CopyAsyncToDevice();
    cache_last_channel_len.CopyAsyncToDevice();

    Ort::IoBinding encoder_binding(encoder_->session);
    encoder_binding.BindInput("input_features", chunk.BindingValue());
    encoder_binding.BindInput("feature_lengths", chunk_length.BindingValue());
    encoder_binding.BindInput("prompt", prompt_buffer.BindingValue());
    encoder_binding.BindInput("att_cache", att_cache.BindingValue());
    encoder_binding.BindInput("conv_cache", conv_cache.BindingValue());
    encoder_binding.BindInput("right_context", right_context.BindingValue());
    encoder_binding.BindInput("cache_last_channel_len", cache_last_channel_len.BindingValue());
    encoder_binding.BindOutput("encoder_states", encoder_states.BindingValue());
    encoder_binding.BindOutput("output_lengths", output_lengths.BindingValue());
    encoder_binding.BindOutput("next_att_cache", next_att_cache.BindingValue());
    encoder_binding.BindOutput("next_conv_cache", next_conv_cache.BindingValue());
    encoder_binding.BindOutput("next_cache_last_channel_len", next_cache_last_channel_len.BindingValue());

    const std::vector<int64_t> frame_shape{1, 1, metadata_.joint_dim};
    Ort::Value encoder_frame;
    std::vector<float> host_frame_storage;
    if (use_device_io)
    {
        encoder_frame =
            Ort::Value::CreateTensor<float>(predict_->DeviceAllocator(), frame_shape.data(), frame_shape.size());
    }
    else
    {
        host_frame_storage.resize(static_cast<size_t>(metadata_.joint_dim));
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        encoder_frame = Ort::Value::CreateTensor<float>(memory, host_frame_storage.data(), host_frame_storage.size(),
                                                        frame_shape.data(), frame_shape.size());
    }

    Ort::IoBinding predict_binding(predict_->session);
    predict_binding.BindInput("input_id", input_id.BindingValue());
    predict_binding.BindInput("hidden", hidden.BindingValue());
    predict_binding.BindInput("cell", cell.BindingValue());
    predict_binding.BindInput("encoder_frame", encoder_frame);
    predict_binding.BindOutput("token_id", token_id.BindingValue());
    predict_binding.BindOutput("next_hidden", next_hidden.BindingValue());
    predict_binding.BindOutput("next_cell", next_cell.BindingValue());

    int64_t base_frame = 0;
    int64_t step = 0;
    for (int64_t start = metadata_.pre_encode_cache_frames_first; start < total_feature_frames;)
    {
        const bool is_first = step == 0;
        const auto chunk_feature_frames =
            is_first ? metadata_.chunk_feature_frames_first : metadata_.chunk_feature_frames_next;
        const auto shift_feature_frames =
            is_first ? metadata_.shift_feature_frames_first : metadata_.shift_feature_frames_next;
        const auto pre_encode_cache_frames =
            is_first ? metadata_.pre_encode_cache_frames_first : metadata_.pre_encode_cache_frames_next;
        const auto input_feature_frames = exported_input_feature_frames;

        const auto current_chunk_length =
            FillStreamingChunk(feature_data, total_feature_frames, feature_bins, start, chunk_feature_frames,
                               pre_encode_cache_frames, input_feature_frames, chunk.HostData());
        *chunk_length.HostData() = static_cast<int32_t>(current_chunk_length);
        RunEncoderStep(*encoder_, encoder_binding, chunk, chunk_length);
        const auto encoded_frames = EncodedFrameCount(metadata_, current_chunk_length);
        GreedyDecodeChunk(*predict_, metadata_, predict_binding, encoder_frame, encoder_states.BindingValue(),
                          encoded_frames, 0, base_frame, input_id, hidden, cell, token_id, next_hidden, next_cell,
                          use_device_io, output);
        att_cache.CopyFrom(next_att_cache);
        conv_cache.CopyFrom(next_conv_cache);
        cache_last_channel_len.CopyFrom(next_cache_last_channel_len);
        base_frame += encoded_frames;
        start += shift_feature_frames;
        step += 1;
    }
}

TranscriptionResult NemotronPipeline::Transcribe(const Audio& audio)
{
    DIN_NVTX_FUNC_RANGE();
    if (audio.sample_rate != metadata_.sampling_rate)
    {
        throw std::runtime_error("expected audio input to match model sample rate");
    }

    const auto transcribe_start = std::chrono::steady_clock::now();
    auto preprocessor_outputs = RunPreprocessor(audio);

    TranscriptionResult result;
    result.audio_seconds = audio.Duration();

    const auto encode_start = std::chrono::steady_clock::now();
    const auto greedy_start = std::chrono::steady_clock::now();
    RunStreamingTranscription(preprocessor_outputs[0], preprocessor_outputs[1], result.generated);
    result.encode_seconds = SecondsSince(encode_start);
    result.greedy_seconds = SecondsSince(greedy_start);
    result.transcribe_seconds = SecondsSince(transcribe_start);
    result.text = tokenizer_->Decode(result.generated.tokens, metadata_.strip_lang_tags);
    return result;
}

TranscriptionResult NemotronPipeline::TranscribeFile(const fs::path& audio_path)
{
    const auto audio = din::io::LoadAudio(audio_path.string(), static_cast<int>(metadata_.sampling_rate));
    return Transcribe(audio);
}

void NemotronPipeline::Print(std::ostream& stream, const TranscriptionResult& result,
                             const TranscriptionOptions& options) const
{
    if (options.timestamps == "none")
    {
        stream << result.text << '\n';
        return;
    }
    if (options.timestamps == "token")
    {
        const auto frame_rate = metadata_.FrameRate();
        for (size_t i = 0; i < result.generated.tokens.size(); ++i)
        {
            stream << result.generated.starts[i] * frame_rate
                   << "s : " << tokenizer_->CleanToken(result.generated.tokens[i]) << " [" << result.generated.tokens[i]
                   << "]\n";
        }
    }
}

}  // namespace din::asr::nemotron
