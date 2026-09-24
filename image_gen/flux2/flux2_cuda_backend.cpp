// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nvtx3/nvtx3.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cu_helper.h"
#include "cuda_kernels.h"
#include "flux2.h"
#include "flux2_runtime_context.h"
#include "ort_session.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

namespace
{
enum class ExecutionProviderMode
{
    Auto,
    Trt,
    Cpu,
};

enum class SamplingBackend
{
    Auto,
    Cuda,
    Cpu,
};

ExecutionProviderMode parse_execution_provider(std::string_view value)
{
    if (value == "auto")
    {
        return ExecutionProviderMode::Auto;
    }
    if (value == "trt")
    {
        return ExecutionProviderMode::Trt;
    }
    if (value == "cpu")
    {
        return ExecutionProviderMode::Cpu;
    }
    throw std::runtime_error("Execution provider must be one of: auto, trt, cpu");
}

SamplingBackend parse_sampling_backend(std::string_view value)
{
    if (value == "auto")
    {
        return SamplingBackend::Auto;
    }
    if (value == "cuda")
    {
        return SamplingBackend::Cuda;
    }
    if (value == "cpu")
    {
        return SamplingBackend::Cpu;
    }
    throw std::runtime_error("Sampling backend must be one of: auto, cuda, cpu");
}

const char* to_string(ExecutionProviderMode mode)
{
    switch (mode)
    {
    case ExecutionProviderMode::Auto:
        return "auto";
    case ExecutionProviderMode::Trt:
        return "trt";
    case ExecutionProviderMode::Cpu:
        return "cpu";
    }
    return "unknown";
}

const char* to_string(SamplingBackend backend)
{
    switch (backend)
    {
    case SamplingBackend::Auto:
        return "auto";
    case SamplingBackend::Cuda:
        return "cuda";
    case SamplingBackend::Cpu:
        return "cpu";
    }
    return "unknown";
}

bool should_run_stage_on_cuda(SamplingBackend backend, bool cuda_available, std::string_view stage_name)
{
    if (backend == SamplingBackend::Cpu)
    {
        return false;
    }
    if (backend == SamplingBackend::Cuda && !cuda_available)
    {
        throw std::runtime_error(std::string(stage_name) +
                                 " CUDA sampling requested but bindings are not device-backed");
    }
    return cuda_available;
}

void fill_position_ids(std::vector<int64_t>& img_ids, std::vector<int64_t>& txt_ids)
{
    for (int64_t b = 0; b < BATCH_SIZE; ++b)
    {
        int64_t idx = 0;
        for (int64_t h = 0; h < LATENT_HEIGHT; ++h)
        {
            for (int64_t w = 0; w < LATENT_WIDTH; ++w, ++idx)
            {
                const int64_t base = (b * IMAGE_SEQUENCE + idx) * 4;
                img_ids[base + 0] = 0;
                img_ids[base + 1] = h;
                img_ids[base + 2] = w;
                img_ids[base + 3] = 0;
            }
        }
    }

    for (int64_t b = 0; b < BATCH_SIZE; ++b)
    {
        for (int64_t t = 0; t < SEQUENCE_LENGTH; ++t)
        {
            const int64_t base = (b * SEQUENCE_LENGTH + t) * 4;
            txt_ids[base + 0] = 0;
            txt_ids[base + 1] = 0;
            txt_ids[base + 2] = 0;
            txt_ids[base + 3] = t;
        }
    }
}

std::vector<float> create_time_schedule()
{
    constexpr double MU = 2.291179894115571;
    std::vector<float> time_schedule(FLOW_STEPS + 1);
    for (int i = 0; i <= FLOW_STEPS; ++i)
    {
        const double t = T_START - (T_START - T_END) * static_cast<double>(i) / FLOW_STEPS;
        time_schedule[i] = static_cast<float>(std::exp(MU) / (std::exp(MU) + std::pow(1.0 / t - 1.0, 1.0)));
        std::cout << "  sigma[" << i << "] = " << time_schedule[i] << std::endl;
    }
    return time_schedule;
}

class EulerStage
{
public:
    void BindInput(const char* name, din::common::TensorBuffer<float>& tensor)
    {
        if (std::string_view{name} == "hidden_states")
        {
            hidden_states_ = &tensor;
        }
        else if (std::string_view{name} == "sample")
        {
            sample_ = &tensor;
        }
        else
        {
            throw std::runtime_error("Unsupported EulerStage input: " + std::string{name});
        }
    }

    void BindOutput(const char* name, din::common::TensorBuffer<float>& tensor)
    {
        if (std::string_view{name} != "hidden_states")
        {
            throw std::runtime_error("Unsupported EulerStage output: " + std::string{name});
        }
        hidden_states_ = &tensor;
    }

    void Run(Ort::SyncStream* stream, SamplingBackend backend, bool device_is_cuda, float t_curr, float t_next)
    {
        if (hidden_states_ == nullptr || sample_ == nullptr)
        {
            throw std::runtime_error("EulerStage bindings are incomplete");
        }

        if (should_run_stage_on_cuda(backend, device_is_cuda, "EulerStage"))
        {
            if (stream == nullptr)
            {
                throw std::runtime_error("EulerStage CUDA path requires an ORT sync stream");
            }
            const size_t hidden_count = hidden_states_->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount();
            launch_flux_euler_kernel(reinterpret_cast<cudaStream_t>(stream->GetHandle()), t_curr, t_next, hidden_count,
                                     hidden_states_->BindingValue().GetTensorMutableData<float>(),
                                     sample_->BindingValue().GetTensorMutableData<float>());
            return;
        }

        if (device_is_cuda)
        {
            auto hidden_ready = hidden_states_->CopyAsyncToHostWithNotification();
            auto sample_ready = sample_->CopyAsyncToHostWithNotification();
            hidden_ready.Sync();
            sample_ready.Sync();
        }

        float* hidden = hidden_states_->HostData();
        float* sample = sample_->HostData();
        const float dt = t_next - t_curr;
        const size_t hidden_count = hidden_states_->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount();
        for (size_t i = 0; i < hidden_count; ++i)
        {
            hidden[i] += dt * sample[i];
        }
        if (device_is_cuda)
        {
            auto hidden_uploaded = hidden_states_->CopyAsyncToDeviceWithNotification();
            hidden_uploaded.Sync();
        }
    }

private:
    din::common::TensorBuffer<float>* hidden_states_ = nullptr;
    din::common::TensorBuffer<float>* sample_ = nullptr;
};

class PostprocessStage
{
public:
    void BindInput(const char* name, din::common::TensorBuffer<float>& tensor)
    {
        if (std::string_view{name} == "hidden_states")
        {
            hidden_states_ = &tensor;
        }
        else if (std::string_view{name} == "scratch")
        {
            scratch_ = &tensor;
        }
        else if (std::string_view{name} == "bn_mean")
        {
            bn_mean_ = &tensor;
        }
        else if (std::string_view{name} == "bn_std")
        {
            bn_std_ = &tensor;
        }
        else
        {
            throw std::runtime_error("Unsupported PostprocessStage input: " + std::string{name});
        }
    }

    void BindOutput(const char* name, din::common::TensorBuffer<float>& tensor)
    {
        if (std::string_view{name} != "latent_sample")
        {
            throw std::runtime_error("Unsupported PostprocessStage output: " + std::string{name});
        }
        decoder_input_ = &tensor;
    }

    void Run(Ort::SyncStream* stream, SamplingBackend backend, bool device_is_cuda)
    {
        if (hidden_states_ == nullptr || scratch_ == nullptr || decoder_input_ == nullptr || bn_mean_ == nullptr ||
            bn_std_ == nullptr)
        {
            throw std::runtime_error("PostprocessStage bindings are incomplete");
        }

        if (should_run_stage_on_cuda(backend, device_is_cuda, "PostprocessStage"))
        {
            if (stream == nullptr)
            {
                throw std::runtime_error("PostprocessStage CUDA path requires an ORT sync stream");
            }
            const size_t postprocess_count = static_cast<size_t>(LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE) *
                                             static_cast<size_t>(LATENT_HEIGHT * PATCH_SIZE) *
                                             static_cast<size_t>(LATENT_WIDTH * PATCH_SIZE);
            launch_flux_postprocess_kernel(
                reinterpret_cast<cudaStream_t>(stream->GetHandle()),
                hidden_states_->BindingValue().GetTensorMutableData<float>(),
                decoder_input_->BindingValue().GetTensorMutableData<float>(),
                bn_mean_->BindingValue().GetTensorMutableData<float>(),
                bn_std_->BindingValue().GetTensorMutableData<float>(), static_cast<unsigned int>(LATENT_CHANNELS),
                static_cast<unsigned int>(LATENT_HEIGHT), static_cast<unsigned int>(LATENT_WIDTH),
                static_cast<unsigned int>(PATCH_SIZE), static_cast<unsigned int>(PATCH_SIZE), postprocess_count);
            return;
        }

        if (device_is_cuda)
        {
            auto hidden_ready = hidden_states_->CopyAsyncToHostWithNotification();
            hidden_ready.Sync();
        }

        float* hidden = hidden_states_->HostData();
        float* scratch = scratch_->HostData();
        float* decoder_input = decoder_input_->HostData();

        constexpr int HW = static_cast<int>(LATENT_HEIGHT * LATENT_WIDTH);
        unpack_latents_with_ids(hidden, scratch, LATENT_HEIGHT, LATENT_WIDTH, LATENT_CHANNELS);
        denormalize_latents(scratch, BN_MEAN, BN_STD, LATENT_CHANNELS, HW);
        unpatchify_latents(scratch, decoder_input, 1, LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE, LATENT_HEIGHT,
                           LATENT_WIDTH, PATCH_SIZE, PATCH_SIZE);
        if (device_is_cuda)
        {
            auto decoder_input_uploaded = decoder_input_->CopyAsyncToDeviceWithNotification();
            decoder_input_uploaded.Sync();
        }
    }

private:
    din::common::TensorBuffer<float>* hidden_states_ = nullptr;
    din::common::TensorBuffer<float>* scratch_ = nullptr;
    din::common::TensorBuffer<float>* decoder_input_ = nullptr;
    din::common::TensorBuffer<float>* bn_mean_ = nullptr;
    din::common::TensorBuffer<float>* bn_std_ = nullptr;
};

struct FluxPipeline
{
    Ort::Session& text_encoder_session;
    Ort::Session& transformer_session;
    Ort::Session& vae_decoder_session;
    Ort::IoBinding& text_encoder_io;
    Ort::IoBinding& transformer_io;
    Ort::IoBinding& vae_decoder_io;
    EulerStage& euler;
    PostprocessStage& postprocess;
    din::common::TensorBuffer<float>& timestep;
};

struct CudaPipelineState
{
    explicit CudaPipelineState(Ort::Env& environment)
        : env(environment)
    {
    }

    Ort::Env& env;
    Ort::ConstEpDevice ep_device{};
    SamplingBackend sampling_backend = SamplingBackend::Cpu;
    bool has_separate_binding = false;
    bool binding_is_cuda = false;
    bool prompt_embeds_valid = false;

    std::optional<Ort::SyncStream> compute_stream;

    std::unique_ptr<din::common::OrtRunner> text_encoder_runner;
    std::unique_ptr<din::common::OrtRunner> transformer_runner;
    std::unique_ptr<din::common::OrtRunner> vae_decoder_runner;

    std::unique_ptr<din::common::TensorBuffer<int64_t>> token;
    std::unique_ptr<din::common::TensorBuffer<int64_t>> attention_mask;
    std::unique_ptr<din::common::TensorBuffer<float>> text_encoder_embeds;
    std::unique_ptr<din::common::TensorBuffer<float>> timestep;
    std::unique_ptr<din::common::TensorBuffer<float>> hidden_states;
    std::unique_ptr<din::common::TensorBuffer<int64_t>> img_ids;
    std::unique_ptr<din::common::TensorBuffer<int64_t>> txt_ids;
    std::unique_ptr<din::common::TensorBuffer<float>> transformer_output;
    std::unique_ptr<din::common::TensorBuffer<float>> decoder_latent;
    std::unique_ptr<din::common::TensorBuffer<float>> decoded_image;
    std::unique_ptr<din::common::TensorBuffer<float>> bn_mean;
    std::unique_ptr<din::common::TensorBuffer<float>> bn_std;

    std::unique_ptr<Ort::IoBinding> text_encoder_io;
    std::unique_ptr<Ort::IoBinding> transformer_io;
    std::unique_ptr<Ort::IoBinding> vae_decoder_io;

    std::vector<float> time_schedule;
    EulerStage euler;
    PostprocessStage postprocess;
};

void run_pipeline(FluxPipeline& pipeline, const std::vector<float>& time_schedule, SamplingBackend sampling_backend,
                  bool device_is_cuda, Ort::SyncStream* compute_stream, bool encode_prompt)
{
    nvtx3::scoped_range nvtx_pipeline("run_pipeline");
    Ort::RunOptions run_options;
    if (compute_stream != nullptr)
    {
        run_options.SetSyncStream(*compute_stream);
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    }

    if (encode_prompt)
    {
        nvtx3::scoped_range nvtx("text_encoder");
        pipeline.text_encoder_session.Run(run_options, pipeline.text_encoder_io);
    }

    for (int step = 0; step < FLOW_STEPS; ++step)
    {
        nvtx3::scoped_range nvtx_step("diffusion_step_" + std::to_string(step));
        const float t_curr = time_schedule[step];
        const float t_next = time_schedule[step + 1];

        pipeline.timestep.HostData()[0] = t_curr;
        if (device_is_cuda)
        {
            auto timestep_uploaded = pipeline.timestep.CopyAsyncToDeviceWithNotification();
            timestep_uploaded.Sync();
        }

        {
            nvtx3::scoped_range nvtx_transformer("transformer");
            pipeline.transformer_session.Run(run_options, pipeline.transformer_io);
        }

        {
            nvtx3::scoped_range nvtx_euler("euler_step");
            pipeline.euler.Run(compute_stream, sampling_backend, device_is_cuda, t_curr, t_next);
        }
    }

    {
        nvtx3::scoped_range nvtx_post("postprocess");
        pipeline.postprocess.Run(compute_stream, sampling_backend, device_is_cuda);
    }

    {
        nvtx3::scoped_range nvtx_vae("vae_decoder");
        pipeline.vae_decoder_session.Run(run_options, pipeline.vae_decoder_io);
    }

    if (compute_stream != nullptr)
    {
        flush_ort_stream(*compute_stream);
    }
}

bool is_trt_rtx_device(Ort::ConstEpDevice ep_device)
{
    return std::string_view{ep_device.EpName()} == kDinNvTensorRTRTXExecutionProvider;
}

bool is_cpu_device(Ort::ConstEpDevice ep_device)
{
    return std::string_view{ep_device.EpName()} == "CPUExecutionProvider";
}

Ort::ConstEpDevice find_cpu_device(Ort::Env& env)
{
    const auto ep_devices = env.GetEpDevices();
    for (const auto& ep_device : ep_devices)
    {
        if (is_cpu_device(ep_device))
        {
            return ep_device;
        }
    }

    throw std::runtime_error("CPUExecutionProvider device was not reported by ONNX Runtime");
}

void initialize_ep(CudaPipelineState& state, Ort::ConstEpDevice ep_device, const Flux2Config& config,
                   ExecutionProviderMode provider_mode, SamplingBackend sampling_backend)
{
    const Flux2ModelPaths model_paths = MakeFlux2ModelPaths(config.model_dir, config.precision);
    const Flux2ModelCachePaths cache_paths = MakeFlux2ModelCachePaths(config.precision);
    const bool trt_rtx_device = is_trt_rtx_device(ep_device);
    const bool cpu_device = is_cpu_device(ep_device);
    const bool has_separate_binding = !cpu_device;
    const bool binding_is_cuda = trt_rtx_device;

    if (sampling_backend == SamplingBackend::Cuda && !binding_is_cuda)
    {
        throw std::runtime_error("CUDA sampling requested but selected EP bindings are not CUDA memory");
    }

    state.ep_device = ep_device;
    state.sampling_backend = sampling_backend;
    state.has_separate_binding = has_separate_binding;
    state.binding_is_cuda = binding_is_cuda;

    if (trt_rtx_device)
    {
        const uint32_t ort_hardware_device_id = ep_device.Device().DeviceId();
        const int cuda_device_ordinal = din::common::ChooseCudaDeviceOrdinal(ep_device);
        CUDA_CHECK(cudaSetDevice(cuda_device_ordinal));

        cudaDeviceProp cuda_device_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&cuda_device_prop, cuda_device_ordinal));
        std::cout << "Using CUDA device ordinal " << cuda_device_ordinal << " (" << cuda_device_prop.name
                  << ") for ORT hardware device_id=" << ort_hardware_device_id << std::endl;

        state.compute_stream.emplace(din::common::CreateTensorRTRTXComputeStream(state.env));
    }

    Ort::SyncStream* compute_stream_ptr = state.compute_stream ? &*state.compute_stream : nullptr;

    std::cout << "=== Loading ONNX Models ===" << std::endl;
    std::string provider;
    if (provider_mode == ExecutionProviderMode::Trt)
    {
        provider = "trt-rtx";
    }
    else
    {
        provider = "cpu";
    }
    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config.ep_context_dir.string();
    const std::string cache_dir = config.ep_cache_dir.string();

    auto make_profile = [](std::string cache_subpath)
    {
        din::common::ModelProfile profile;
        profile.cache_subpath = std::move(cache_subpath);
        profile.enable_cuda_graph = false;
        profile.embed_ep_context = false;
        profile.extra_ep_options.emplace_back("nv_length_aux_stream_array", "0");
        return profile;
    };

    state.text_encoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.text_encoder_model.string(), provider, cache_dir, ep_context,
        make_profile(cache_paths.text_encoder), compute_stream_ptr);
    std::cout << "  Text encoder loaded" << std::endl;
    state.transformer_runner =
        std::make_unique<din::common::OrtRunner>(state.env, model_paths.transformer_model.string(), provider, cache_dir,
                                                 ep_context, make_profile(cache_paths.transformer), compute_stream_ptr);
    std::cout << "  Transformer loaded" << std::endl;
    state.vae_decoder_runner =
        std::make_unique<din::common::OrtRunner>(state.env, model_paths.vae_decoder_model.string(), provider, cache_dir,
                                                 ep_context, make_profile(cache_paths.vae_decoder), compute_stream_ptr);
    std::cout << "  VAE decoder loaded" << std::endl;

    std::vector<int64_t> token_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    state.token = std::make_unique<din::common::TensorBuffer<int64_t>>(*state.text_encoder_runner, token_shape,
                                                                       has_separate_binding);

    std::vector<int64_t> attn_mask_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    state.attention_mask = std::make_unique<din::common::TensorBuffer<int64_t>>(*state.text_encoder_runner,
                                                                                attn_mask_shape, has_separate_binding);

    std::vector<int64_t> te_out_shape = {BATCH_SIZE, SEQUENCE_LENGTH, TEXT_ENCODER_EMBED_DIM};
    state.text_encoder_embeds = std::make_unique<din::common::TensorBuffer<float>>(*state.text_encoder_runner,
                                                                                   te_out_shape, has_separate_binding);

    std::vector<int64_t> timestep_shape = {BATCH_SIZE};
    state.timestep = std::make_unique<din::common::TensorBuffer<float>>(*state.transformer_runner, timestep_shape,
                                                                        has_separate_binding);

    std::vector<int64_t> hidden_shape = {BATCH_SIZE, TRANSFORMER_HIDDEN_DIM, LATENT_CHANNELS};
    state.hidden_states = std::make_unique<din::common::TensorBuffer<float>>(*state.transformer_runner, hidden_shape,
                                                                             has_separate_binding);

    std::vector<int64_t> img_ids_shape = {BATCH_SIZE, IMAGE_SEQUENCE, 4};
    state.img_ids = std::make_unique<din::common::TensorBuffer<int64_t>>(*state.transformer_runner, img_ids_shape,
                                                                         has_separate_binding);

    std::vector<int64_t> txt_ids_shape = {BATCH_SIZE, SEQUENCE_LENGTH, 4};
    state.txt_ids = std::make_unique<din::common::TensorBuffer<int64_t>>(*state.transformer_runner, txt_ids_shape,
                                                                         has_separate_binding);

    state.transformer_output = std::make_unique<din::common::TensorBuffer<float>>(*state.transformer_runner,
                                                                                  hidden_shape, has_separate_binding);

    std::vector<int64_t> dec_latent_shape = {BATCH_SIZE, LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE,
                                             LATENT_HEIGHT * PATCH_SIZE, LATENT_WIDTH * PATCH_SIZE};
    state.decoder_latent = std::make_unique<din::common::TensorBuffer<float>>(*state.vae_decoder_runner,
                                                                              dec_latent_shape, has_separate_binding);

    std::vector<int64_t> image_shape = {BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH};
    state.decoded_image = std::make_unique<din::common::TensorBuffer<float>>(*state.vae_decoder_runner, image_shape,
                                                                             has_separate_binding);

    std::vector<int64_t> bn_shape = {LATENT_CHANNELS};
    state.bn_mean =
        std::make_unique<din::common::TensorBuffer<float>>(*state.transformer_runner, bn_shape, has_separate_binding);
    state.bn_std =
        std::make_unique<din::common::TensorBuffer<float>>(*state.transformer_runner, bn_shape, has_separate_binding);

    state.text_encoder_io = std::make_unique<Ort::IoBinding>(state.text_encoder_runner->session);
    state.text_encoder_io->BindInput("input_ids", state.token->BindingValue());
    state.text_encoder_io->BindInput("attention_mask", state.attention_mask->BindingValue());
    state.text_encoder_io->BindOutput("prompt_embeds", state.text_encoder_embeds->BindingValue());

    state.transformer_io = std::make_unique<Ort::IoBinding>(state.transformer_runner->session);
    state.transformer_io->BindInput("hidden_states", state.hidden_states->BindingValue());
    state.transformer_io->BindInput("encoder_hidden_states", state.text_encoder_embeds->BindingValue());
    state.transformer_io->BindInput("timestep", state.timestep->BindingValue());
    state.transformer_io->BindInput("img_ids", state.img_ids->BindingValue());
    state.transformer_io->BindInput("txt_ids", state.txt_ids->BindingValue());
    state.transformer_io->BindOutput("sample", state.transformer_output->BindingValue());

    state.vae_decoder_io = std::make_unique<Ort::IoBinding>(state.vae_decoder_runner->session);
    state.vae_decoder_io->BindInput("latent_sample", state.decoder_latent->BindingValue());
    state.vae_decoder_io->BindOutput("sample", state.decoded_image->BindingValue());

    std::vector<int64_t> img_ids_cpu(state.img_ids->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount());
    std::vector<int64_t> txt_ids_cpu(state.txt_ids->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount());
    fill_position_ids(img_ids_cpu, txt_ids_cpu);
    std::copy(img_ids_cpu.begin(), img_ids_cpu.end(), state.img_ids->HostData());
    std::copy(txt_ids_cpu.begin(), txt_ids_cpu.end(), state.txt_ids->HostData());

    const Flux2TextEncoderInputs text_inputs = TokenizeFlux2Prompt(model_paths, config.prompt);
    FillTextEncoderInputs(text_inputs.token_ids, text_inputs.pad_token_id, state.token->HostData(),
                          state.attention_mask->HostData(), BATCH_SIZE, SEQUENCE_LENGTH);
    std::copy_n(BN_MEAN, LATENT_CHANNELS, state.bn_mean->HostData());
    std::copy_n(BN_STD, LATENT_CHANNELS, state.bn_std->HostData());

    if (has_separate_binding)
    {
        auto img_ids_ready = state.img_ids->CopyAsyncToDeviceWithNotification();
        auto txt_ids_ready = state.txt_ids->CopyAsyncToDeviceWithNotification();
        auto token_ready = state.token->CopyAsyncToDeviceWithNotification();
        auto attention_mask_ready = state.attention_mask->CopyAsyncToDeviceWithNotification();
        auto bn_mean_ready = state.bn_mean->CopyAsyncToDeviceWithNotification();
        auto bn_std_ready = state.bn_std->CopyAsyncToDeviceWithNotification();
        img_ids_ready.Sync();
        txt_ids_ready.Sync();
        token_ready.Sync();
        attention_mask_ready.Sync();
        bn_mean_ready.Sync();
        bn_std_ready.Sync();
    }

    state.time_schedule = create_time_schedule();

    state.euler.BindInput("hidden_states", *state.hidden_states);
    state.euler.BindInput("sample", *state.transformer_output);
    state.euler.BindOutput("hidden_states", *state.hidden_states);

    state.postprocess.BindInput("hidden_states", *state.hidden_states);
    state.postprocess.BindInput("scratch", *state.transformer_output);
    state.postprocess.BindInput("bn_mean", *state.bn_mean);
    state.postprocess.BindInput("bn_std", *state.bn_std);
    state.postprocess.BindOutput("latent_sample", *state.decoder_latent);
}

Flux2Image run_initialized_pipeline(CudaPipelineState& state, unsigned int seed)
{
    Ort::SyncStream* compute_stream_ptr = state.compute_stream ? &*state.compute_stream : nullptr;

    FluxPipeline pipeline{state.text_encoder_runner->session,
                          state.transformer_runner->session,
                          state.vae_decoder_runner->session,
                          *state.text_encoder_io,
                          *state.transformer_io,
                          *state.vae_decoder_io,
                          state.euler,
                          state.postprocess,
                          *state.timestep};

    {
        nvtx3::scoped_range nvtx("initialize_latents");
        std::cout << "\n=== Initializing Latent (seed=" << seed << ") ===" << std::endl;
        const size_t hidden_count = state.hidden_states->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount();
        initialize_latent(state.hidden_states->HostData(), hidden_count, seed);
        if (state.binding_is_cuda)
        {
            auto hidden_uploaded = state.hidden_states->CopyAsyncToDeviceWithNotification();
            hidden_uploaded.Sync();
        }
    }

    run_pipeline(pipeline, state.time_schedule, state.sampling_backend, state.binding_is_cuda, compute_stream_ptr,
                 !state.prompt_embeds_valid);
    state.prompt_embeds_valid = true;

    Flux2Image image;
    image.height = static_cast<int>(IMAGE_HEIGHT);
    image.width = static_cast<int>(IMAGE_WIDTH);
    image.data.resize(state.decoded_image->BindingValue().GetTensorTypeAndShapeInfo().GetElementCount());
    if (state.binding_is_cuda)
    {
        auto image_ready = state.decoded_image->CopyAsyncToHostWithNotification();
        image_ready.Sync();
    }
    const float* image_data = state.decoded_image->HostData();
    std::copy_n(image_data, image.data.size(), image.data.data());
    return image;
}

class CudaFlux2ProcessingPipeline final : public Flux2ProcessingPipeline
{
public:
    CudaFlux2ProcessingPipeline(Flux2Config config, Flux2RuntimeContext& runtime)
        : config_(std::move(config))
        , runtime_(runtime)
    {
    }

    void Initialize() override
    {
        if (state_)
        {
            return;
        }

        const ExecutionProviderMode provider =
            config_.provider == Flux2ExecutionProvider::Cpu ? ExecutionProviderMode::Cpu : ExecutionProviderMode::Trt;
        const SamplingBackend sampling_backend =
            config_.processing == Flux2ProcessingBackend::Cpu ? SamplingBackend::Cpu : SamplingBackend::Cuda;

        if (provider == ExecutionProviderMode::Cpu && sampling_backend == SamplingBackend::Cuda)
        {
            throw std::runtime_error("CUDA processing requires --provider trt-rtx");
        }

        std::cout << "Model dir: " << config_.model_dir.string() << "\n"
                  << "Execution provider: " << to_string(provider) << "\n"
                  << "Sampling backend: " << to_string(sampling_backend) << "\n"
                  << std::endl;

        state_ = std::make_unique<CudaPipelineState>(runtime_.env);
        if (provider == ExecutionProviderMode::Cpu)
        {
            std::cout << "Using CPU execution provider" << std::endl;
            initialize_ep(*state_, find_cpu_device(state_->env), config_, provider, sampling_backend);
            return;
        }

        const auto trt_device = runtime_.trt_device;
        if (std::strcmp(trt_device.EpName(), kDinNvTensorRTRTXExecutionProvider) != 0)
        {
            throw std::runtime_error("TensorRT RTX execution provider requested but no device was found");
        }

        std::cout << "Using TensorRT RTX execution provider" << std::endl;
        initialize_ep(*state_, trt_device, config_, provider, sampling_backend);
    }

    void SetPrompt(std::string prompt) override
    {
        config_.prompt = std::move(prompt);
        if (!state_)
        {
            return;
        }
        state_->prompt_embeds_valid = false;

        const Flux2ModelPaths model_paths = MakeFlux2ModelPaths(config_.model_dir, config_.precision);
        const Flux2TextEncoderInputs text_inputs = TokenizeFlux2Prompt(model_paths, config_.prompt);
        FillTextEncoderInputs(text_inputs.token_ids, text_inputs.pad_token_id, state_->token->HostData(),
                              state_->attention_mask->HostData(), BATCH_SIZE, SEQUENCE_LENGTH);

        if (state_->has_separate_binding)
        {
            auto token_ready = state_->token->CopyAsyncToDeviceWithNotification();
            auto attention_mask_ready = state_->attention_mask->CopyAsyncToDeviceWithNotification();
            token_ready.Sync();
            attention_mask_ready.Sync();
        }
    }

    Flux2Image GenerateImage(unsigned int seed) override
    {
        if (!state_)
        {
            throw std::runtime_error("Flux2 CUDA pipeline was not initialized");
        }
        return run_initialized_pipeline(*state_, seed);
    }

private:
    Flux2Config config_;
    Flux2RuntimeContext& runtime_;
    std::unique_ptr<CudaPipelineState> state_;
};
}  // namespace

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2CudaPipeline(const Flux2Config& config,
                                                                 Flux2RuntimeContext& runtime)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    return std::make_unique<CudaFlux2ProcessingPipeline>(config, runtime);
}
