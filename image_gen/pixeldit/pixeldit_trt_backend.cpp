// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PixelDiT pipeline on TensorRT RTX (GPU). The text encoder and DiT always run in TensorRT RTX; --processing selects
// where the sampling math (CFG + flow DPM-Solver++ update) runs. "cpu": on the host, with the velocity downloaded and
// the sample uploaded around every DiT call. "cuda": kernels (cuda_kernels.cu) on the device buffers, all on the
// TensorRT RTX compute stream, with one host sync per image (like image_gen/flux2/flux2_cuda_backend.cpp).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda_kernels.h"
#include "nvtx_helper.h"
#include "ort_session.h"
#include "pixeldit.h"
#include "pixeldit_runtime_context.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

namespace din::image_gen
{
namespace
{

#define PIXELDIT_CUDA_CHECK(call)                                                                              \
    do                                                                                                         \
    {                                                                                                          \
        const cudaError_t _err = (call);                                                                       \
        if (_err != cudaSuccess)                                                                               \
        {                                                                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) + " at " #call); \
        }                                                                                                      \
    } while (0)

enum class SamplingBackend
{
    Cpu,
    Cuda,
};

const char* to_string(SamplingBackend backend)
{
    switch (backend)
    {
    case SamplingBackend::Cpu:
        return "cpu";
    case SamplingBackend::Cuda:
        return "cuda";
    }
    return "unknown";
}

using FloatBuffer = din::common::TensorBuffer<float>;
using Int64Buffer = din::common::TensorBuffer<int64_t>;

template <typename T>
size_t element_count(din::common::TensorBuffer<T>& buffer)
{
    return buffer.BindingValue().GetTensorTypeAndShapeInfo().GetElementCount();
}

// Pointer to the bound tensor of a device-backed buffer: device memory, or pinned host memory on unified-memory GPUs.
float* device_data(FloatBuffer& buffer)
{
    return buffer.BindingValue().GetTensorMutableData<float>();
}

// cudaMemcpyDefault so the copy also works when the bound tensors are pinned host memory (unified memory).
void copy_device_async(float* dst, const float* src, size_t count, cudaStream_t stream)
{
    PIXELDIT_CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDefault, stream));
}

template <typename T>
void write_raw(const std::filesystem::path& path, const T* data, size_t count)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Failed to open dump file " + path.string());
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

std::vector<float> read_raw_floats(const std::filesystem::path& path, size_t expected_count)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        throw std::runtime_error("Failed to open --init-noise file " + path.string());
    }
    const auto bytes = static_cast<size_t>(in.tellg());
    if (bytes != expected_count * sizeof(float))
    {
        throw std::runtime_error("--init-noise file " + path.string() + " has " + std::to_string(bytes) +
                                 " bytes, expected " + std::to_string(expected_count * sizeof(float)) +
                                 " (fp32 [1, 3, H, W])");
    }
    std::vector<float> values(expected_count);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    return values;
}

std::vector<int64_t> session_shape(Ort::Session& session, const char* name, bool input)
{
    const size_t count = input ? session.GetInputCount() : session.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < count; ++i)
    {
        const auto io_name = input ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        if (std::string_view{io_name.get()} == name)
        {
            const auto type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
            return type_info.GetTensorTypeAndShapeInfo().GetShape();
        }
    }
    throw std::runtime_error(std::string("ONNX model has no ") + (input ? "input " : "output ") + name);
}

std::string shape_string(const std::vector<int64_t>& shape)
{
    std::ostringstream text;
    text << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        text << (i ? ", " : "") << shape[i];
    }
    text << "]";
    return text.str();
}

// CFG + data prediction: velocity [2, n] ([negative, positive]) and the current sample -> x0 estimate.
class CfgDataPredictionStage
{
public:
    void BindInput(const char* name, FloatBuffer& tensor)
    {
        if (std::string_view{name} == "velocity")
        {
            velocity_ = &tensor;
        }
        else if (std::string_view{name} == "sample")
        {
            sample_ = &tensor;
        }
        else
        {
            throw std::runtime_error("Unsupported CfgDataPredictionStage input: " + std::string{name});
        }
    }

    // cpu: the velocity must already be on the host. cuda: everything stays on the device, ordered on `stream`.
    void Run(cudaStream_t stream, SamplingBackend backend, float t, float cfg_scale, FloatBuffer& x0)
    {
        if (velocity_ == nullptr || sample_ == nullptr)
        {
            throw std::runtime_error("CfgDataPredictionStage bindings are incomplete");
        }
        const size_t n = element_count(*sample_);
        if (backend == SamplingBackend::Cuda)
        {
            launch_pixeldit_cfg_x0_kernel(stream, device_data(*velocity_), device_data(*sample_), t, cfg_scale,
                                          device_data(x0), n);
            return;
        }
        cfg_data_prediction(velocity_->HostData(), sample_->HostData(), t, cfg_scale, x0.HostData(), n);
    }

private:
    FloatBuffer* velocity_ = nullptr;
    FloatBuffer* sample_ = nullptr;
};

// Flow DPM-Solver++ update of the sample from t_prev_0 to t (1st or 2nd order).
class DpmSolverStage
{
public:
    void BindOutput(const char* name, FloatBuffer& tensor)
    {
        if (std::string_view{name} != "sample")
        {
            throw std::runtime_error("Unsupported DpmSolverStage output: " + std::string{name});
        }
        sample_ = &tensor;
    }

    void Run(cudaStream_t stream, SamplingBackend backend, int order, FloatBuffer* model_prev_1,
             FloatBuffer& model_prev_0, float t_prev_1, float t_prev_0, float t)
    {
        if (sample_ == nullptr)
        {
            throw std::runtime_error("DpmSolverStage bindings are incomplete");
        }
        const size_t n = element_count(*sample_);
        if (backend == SamplingBackend::Cuda)
        {
            if (order == 1 && t == 0.0f)
            {
                // Final step: the update is exactly x0 (see dpm_first_order_update).
                copy_device_async(device_data(*sample_), device_data(model_prev_0), n, stream);
            }
            else if (order == 1)
            {
                const DpmFirstOrderCoeffs k = dpm_first_order_coeffs(t_prev_0, t);
                launch_pixeldit_dpm1_kernel(stream, device_data(*sample_), device_data(model_prev_0), k.ratio, k.coeff,
                                            n);
            }
            else
            {
                const DpmSecondOrderCoeffs k = dpm_second_order_coeffs(t_prev_1, t_prev_0, t);
                launch_pixeldit_dpm2_kernel(stream, device_data(*sample_), device_data(*model_prev_1),
                                            device_data(model_prev_0), k.ratio, k.coeff, k.half_coeff, k.inv_r0, n);
            }
            return;
        }
        if (order == 1)
        {
            dpm_first_order_update(sample_->HostData(), model_prev_0.HostData(), t_prev_0, t, n);
        }
        else
        {
            dpm_second_order_update(sample_->HostData(), model_prev_1->HostData(), model_prev_0.HostData(), t_prev_1,
                                    t_prev_0, t, n);
        }
    }

private:
    FloatBuffer* sample_ = nullptr;
};

struct SamplerSettings
{
    int steps = 0;
    float cfg_scale = 0.0f;
    float flow_shift = 0.0f;
    std::string negative_prompt;
};

// Per-image wall time of each sampling stage with --processing cpu (host view; GPU work is inside `dit`). With
// --processing cuda the host only enqueues work, so only the total sampling time is reported.
struct StageTiming
{
    std::chrono::duration<double> upload{};  // host packing + host->device copies
    std::chrono::duration<double> dit{};     // DiT run + velocity device->host copy
    std::chrono::duration<double> cfg{};     // host CFG + x0
    std::chrono::duration<double> solver{};  // host DPM-Solver++ update
    int evaluations = 0;
};

struct PipelineState
{
    explicit PipelineState(Ort::Env& environment)
        : env(environment)
    {
    }

    Ort::Env& env;
    PixelDiTModelPaths model_paths;
    PixelDiTPipelineConfig pipeline_config;
    SamplerSettings sampler;
    SamplingBackend sampling_backend = SamplingBackend::Cpu;
    int64_t text_hidden = 0;
    std::unique_ptr<din::io::Tokenizer> tokenizer;
    bool prompt_embeds_valid = false;
    std::string prompt;

    std::optional<Ort::SyncStream> compute_stream;
    std::unique_ptr<din::common::OrtRunner> text_encoder_runner;
    std::unique_ptr<din::common::OrtRunner> transformer_runner;

    // Text encoder IO (device-backed with pinned host mirrors).
    std::unique_ptr<Int64Buffer> input_ids;
    std::unique_ptr<Int64Buffer> attention_mask;
    std::unique_ptr<Int64Buffer> select_index;
    std::unique_ptr<FloatBuffer> prompt_embeds;
    // DiT IO.
    std::unique_ptr<FloatBuffer> dit_hidden_states;
    std::unique_ptr<FloatBuffer> dit_timestep;
    std::unique_ptr<FloatBuffer> dit_encoder_hidden_states;
    std::unique_ptr<FloatBuffer> dit_velocity;
    // Sampling state: current sample and the two most recent x0 estimates. Host-only with --processing cpu,
    // device-backed with --processing cuda.
    std::unique_ptr<FloatBuffer> sample;
    std::unique_ptr<FloatBuffer> model_a;
    std::unique_ptr<FloatBuffer> model_b;
    // --processing cuda only: DiT timestep for every model evaluation, [steps, dit_batch], uploaded once per image
    // and copied into dit_timestep on the device (rewriting the pinned dit_timestep mirror each step would race the
    // still-queued copies of earlier steps).
    std::unique_ptr<FloatBuffer> timestep_table;

    std::unique_ptr<Ort::IoBinding> text_encoder_io;
    std::unique_ptr<Ort::IoBinding> transformer_io;

    CfgDataPredictionStage cfg;
    DpmSolverStage solver;
    StageTiming timing;
};

Ort::RunOptions make_run_options(PipelineState& state)
{
    Ort::RunOptions run_options;
    if (state.compute_stream)
    {
        run_options.SetSyncStream(*state.compute_stream);
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    }
    return run_options;
}

void initialize_state(PipelineState& state, Ort::ConstEpDevice trt_device, const PixelDiTConfig& config)
{
    state.model_paths = MakePixelDiTModelPaths(config.model_dir);
    state.pipeline_config = LoadPixelDiTPipelineConfig(state.model_paths.pipeline_config);
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    if (!pc.file_found)
    {
        std::cout << "pipeline_config.json not found at " << state.model_paths.pipeline_config.string()
                  << "; using built-in defaults (1024x1024 export)" << std::endl;
    }
    else if (!pc.defaults_used.empty())
    {
        std::cout << "pipeline_config.json is missing fields, using defaults for:";
        for (const auto& name : pc.defaults_used)
        {
            std::cout << " " << name;
        }
        std::cout << std::endl;
    }

    state.sampler.steps = config.steps.value_or(pc.steps);
    state.sampler.cfg_scale = config.cfg_scale.value_or(pc.cfg_scale);
    state.sampler.flow_shift = config.flow_shift.value_or(pc.flow_shift);
    state.sampler.negative_prompt = config.negative_prompt.value_or(pc.negative_prompt);
    if (state.sampler.steps < pc.order)
    {
        throw std::invalid_argument("--steps must be at least " + std::to_string(pc.order));
    }
    std::cout << "Image size: " << pc.height << "x" << pc.width << "\n"
              << "Steps: " << state.sampler.steps << ", CFG scale: " << state.sampler.cfg_scale
              << ", flow shift: " << state.sampler.flow_shift << "\n"
              << "Negative prompt: " << state.sampler.negative_prompt << std::endl;

    std::cout << "Loading tokenizer " << state.model_paths.tokenizer_json.string() << std::endl;
    state.tokenizer = std::make_unique<din::io::Tokenizer>(state.model_paths.tokenizer_json.string(),
                                                           din::io::TokenizerFormat::Json);
    const size_t chi_tokens = state.tokenizer->Encode(pc.chi_prompt, true).size();
    if (static_cast<int64_t>(chi_tokens) + pc.txt_max_length - 2 != pc.text_seq_len)
    {
        throw std::runtime_error("chi_prompt tokenizes to " + std::to_string(chi_tokens) +
                                 " tokens, inconsistent with text_seq_len=" + std::to_string(pc.text_seq_len) +
                                 " and txt_max_length=" + std::to_string(pc.txt_max_length));
    }

    const int cuda_device_ordinal = din::common::ChooseCudaDeviceOrdinal(trt_device, "PIXELDIT_CUDA_DEVICE_ID");
    PIXELDIT_CUDA_CHECK(cudaSetDevice(cuda_device_ordinal));
    cudaDeviceProp cuda_device_prop{};
    PIXELDIT_CUDA_CHECK(cudaGetDeviceProperties(&cuda_device_prop, cuda_device_ordinal));
    std::cout << "TensorRT RTX GPU: " << cuda_device_prop.name << " (CUDA device ordinal " << cuda_device_ordinal
              << ")" << std::endl;
    state.compute_stream.emplace(din::common::CreateTensorRTRTXComputeStream(state.env));

    std::cout << "=== Loading ONNX Models ===" << std::endl;
    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config.ep_context_dir.string();
    const std::string cache_dir = config.ep_cache_dir.string();
    const auto make_profile = [](std::string cache_subpath)
    {
        din::common::ModelProfile profile;
        profile.cache_subpath = std::move(cache_subpath);
        profile.enable_cuda_graph = false;
        profile.embed_ep_context = false;  // both engines exceed the ~2 GB protobuf limit
        // Leave TensorRT auxiliary streams at the EP default: forcing 0 (as Flux2 does) made the 1024x1024 DiT
        // call take ~67 ms instead of ~48 ms on an RTX PRO 6000.
        return profile;
    };
    Ort::SyncStream* stream = &*state.compute_stream;
    state.text_encoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, state.model_paths.text_encoder_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile("text_encoder"), stream);
    std::cout << "  Text encoder loaded" << std::endl;
    state.transformer_runner = std::make_unique<din::common::OrtRunner>(
        state.env, state.model_paths.transformer_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile("transformer"), stream);
    std::cout << "  Transformer loaded" << std::endl;

    Ort::Session& te = state.text_encoder_runner->session;
    Ort::Session& dit = state.transformer_runner->session;
    const auto te_out = session_shape(te, "prompt_embeds", false);
    if (te_out.size() != 3)
    {
        throw std::runtime_error("text_encoder prompt_embeds must be rank 3, got " + shape_string(te_out));
    }
    state.text_hidden = te_out[2];
    const std::vector<int64_t> text_shape = {1, pc.text_seq_len};
    const std::vector<int64_t> select_shape = {pc.txt_max_length};
    const std::vector<int64_t> embeds_shape = {1, pc.txt_max_length, state.text_hidden};
    const std::vector<int64_t> image_shape = {pc.dit_batch, 3, pc.height, pc.width};
    const std::vector<int64_t> timestep_shape = {pc.dit_batch};
    const std::vector<int64_t> dit_text_shape = {pc.dit_batch, pc.txt_max_length, state.text_hidden};
    const std::vector<int64_t> sample_shape = {1, 3, pc.height, pc.width};
    auto& te_runner = *state.text_encoder_runner;
    auto& dit_runner = *state.transformer_runner;
    state.input_ids = std::make_unique<Int64Buffer>(te_runner, text_shape, true);
    state.attention_mask = std::make_unique<Int64Buffer>(te_runner, text_shape, true);
    state.select_index = std::make_unique<Int64Buffer>(te_runner, select_shape, true);
    state.prompt_embeds = std::make_unique<FloatBuffer>(te_runner, embeds_shape, true);
    state.dit_hidden_states = std::make_unique<FloatBuffer>(dit_runner, image_shape, true);
    state.dit_timestep = std::make_unique<FloatBuffer>(dit_runner, timestep_shape, true);
    state.dit_encoder_hidden_states = std::make_unique<FloatBuffer>(dit_runner, dit_text_shape, true);
    state.dit_velocity = std::make_unique<FloatBuffer>(dit_runner, image_shape, true);
    const bool sample_on_device = state.sampling_backend == SamplingBackend::Cuda;
    state.sample = std::make_unique<FloatBuffer>(dit_runner, sample_shape, sample_on_device);
    state.model_a = std::make_unique<FloatBuffer>(dit_runner, sample_shape, sample_on_device);
    state.model_b = std::make_unique<FloatBuffer>(dit_runner, sample_shape, sample_on_device);
    if (sample_on_device)
    {
        const std::vector<int64_t> table_shape = {state.sampler.steps, pc.dit_batch};
        state.timestep_table = std::make_unique<FloatBuffer>(dit_runner, table_shape, true);
    }

    state.text_encoder_io = std::make_unique<Ort::IoBinding>(te);
    state.text_encoder_io->BindInput("input_ids", state.input_ids->BindingValue());
    state.text_encoder_io->BindInput("attention_mask", state.attention_mask->BindingValue());
    state.text_encoder_io->BindInput("select_index", state.select_index->BindingValue());
    state.text_encoder_io->BindOutput("prompt_embeds", state.prompt_embeds->BindingValue());

    state.transformer_io = std::make_unique<Ort::IoBinding>(dit);
    state.transformer_io->BindInput("hidden_states", state.dit_hidden_states->BindingValue());
    state.transformer_io->BindInput("timestep", state.dit_timestep->BindingValue());
    state.transformer_io->BindInput("encoder_hidden_states", state.dit_encoder_hidden_states->BindingValue());
    state.transformer_io->BindOutput("velocity", state.dit_velocity->BindingValue());

    state.cfg.BindInput("velocity", *state.dit_velocity);
    state.cfg.BindInput("sample", *state.sample);
    state.solver.BindOutput("sample", *state.sample);
}

// Encodes one prompt and writes it into row `row` of the DiT encoder_hidden_states host buffer.
void encode_prompt(PipelineState& state, const std::string& text, const std::vector<int64_t>& select, int64_t row,
                   const std::filesystem::path& dump_path)
{
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    const PixelDiTTextInputs inputs = MakeTextEncoderInputs(*state.tokenizer, text, pc.text_seq_len, pc.pad_token_id);
    if (inputs.num_tokens > static_cast<size_t>(pc.text_seq_len))
    {
        std::cout << "Prompt tokens truncated from " << inputs.num_tokens << " to " << pc.text_seq_len << std::endl;
    }
    std::copy(inputs.input_ids.begin(), inputs.input_ids.end(), state.input_ids->HostData());
    std::copy(inputs.attention_mask.begin(), inputs.attention_mask.end(), state.attention_mask->HostData());
    std::copy(select.begin(), select.end(), state.select_index->HostData());
    if (!dump_path.empty())
    {
        write_raw(dump_path, inputs.input_ids.data(), inputs.input_ids.size());
    }

    auto ids_ready = state.input_ids->CopyAsyncToDeviceWithNotification();
    auto mask_ready = state.attention_mask->CopyAsyncToDeviceWithNotification();
    auto select_ready = state.select_index->CopyAsyncToDeviceWithNotification();
    ids_ready.Sync();
    mask_ready.Sync();
    select_ready.Sync();

    Ort::RunOptions run_options = make_run_options(state);
    state.text_encoder_runner->session.Run(run_options, *state.text_encoder_io);
    auto embeds_ready = state.prompt_embeds->CopyAsyncToHostWithNotification();
    embeds_ready.Sync();

    const size_t row_size = static_cast<size_t>(pc.txt_max_length * state.text_hidden);
    std::copy_n(state.prompt_embeds->HostData(), row_size,
                state.dit_encoder_hidden_states->HostData() + static_cast<size_t>(row) * row_size);
}

void encode_prompts(PipelineState& state, const PixelDiTConfig& config)
{
    din::common::nvtx_scoped_range nvtx("encode_prompts");
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    const bool dump = !config.dump_dir.empty();
    const auto start = std::chrono::steady_clock::now();
    // Batch row 0 is the negative prompt, row 1 the positive prompt (chi_prompt prefix + prompt).
    encode_prompt(state, state.sampler.negative_prompt, pc.negative_select_index, 0,
                  dump ? config.dump_dir / "tokens_negative.bin" : std::filesystem::path{});
    encode_prompt(state, pc.chi_prompt + StripWhitespace(state.prompt), pc.positive_select_index, 1,
                  dump ? config.dump_dir / "tokens_positive.bin" : std::filesystem::path{});
    auto uploaded = state.dit_encoder_hidden_states->CopyAsyncToDeviceWithNotification();
    uploaded.Sync();
    if (dump)
    {
        write_raw(config.dump_dir / "prompt_embeds.bin", state.dit_encoder_hidden_states->HostData(),
                  element_count(*state.dit_encoder_hidden_states));
    }
    state.prompt_embeds_valid = true;
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Text encoding (negative + positive): " << elapsed.count() << " s" << std::endl;
}

cudaStream_t compute_stream(PipelineState& state)
{
    return reinterpret_cast<cudaStream_t>(state.compute_stream->GetHandle());
}

// --processing cuda: one model evaluation, enqueued on the compute stream without any host sync.
// `eval_index` selects the row of timestep_table (timestep t * timestep_scale).
void data_prediction_cuda(PipelineState& state, float t, int eval_index, FloatBuffer& x0)
{
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    const size_t n = element_count(*state.sample);
    const cudaStream_t stream = compute_stream(state);
    float* hidden = device_data(*state.dit_hidden_states);
    copy_device_async(hidden, device_data(*state.sample), n, stream);
    copy_device_async(hidden + n, device_data(*state.sample), n, stream);
    const size_t batch = static_cast<size_t>(pc.dit_batch);
    copy_device_async(device_data(*state.dit_timestep),
                      device_data(*state.timestep_table) + static_cast<size_t>(eval_index) * batch, batch, stream);
    {
        din::common::nvtx_scoped_range nvtx("transformer");
        Ort::RunOptions run_options = make_run_options(state);
        state.transformer_runner->session.Run(run_options, *state.transformer_io);
    }
    {
        din::common::nvtx_scoped_range nvtx("cfg_data_prediction");
        state.cfg.Run(stream, SamplingBackend::Cuda, t, state.sampler.cfg_scale, x0);
    }
    ++state.timing.evaluations;
}

// One model evaluation: DiT on [sample, sample] at time t, then CFG -> x0 estimate.
void data_prediction(PipelineState& state, float t, int eval_index, din::common::TensorBuffer<float>& x0)
{
    if (state.sampling_backend == SamplingBackend::Cuda)
    {
        data_prediction_cuda(state, t, eval_index, x0);
        return;
    }
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    const size_t n = element_count(*state.sample);
    const auto start = std::chrono::steady_clock::now();
    float* hidden = state.dit_hidden_states->HostData();
    std::copy_n(state.sample->HostData(), n, hidden);
    std::copy_n(state.sample->HostData(), n, hidden + n);
    const float timestep = t * pc.timestep_scale;
    std::fill_n(state.dit_timestep->HostData(), static_cast<size_t>(pc.dit_batch), timestep);
    auto hidden_ready = state.dit_hidden_states->CopyAsyncToDeviceWithNotification();
    auto timestep_ready = state.dit_timestep->CopyAsyncToDeviceWithNotification();
    hidden_ready.Sync();
    timestep_ready.Sync();
    const auto uploaded = std::chrono::steady_clock::now();

    {
        din::common::nvtx_scoped_range nvtx("transformer");
        Ort::RunOptions run_options = make_run_options(state);
        state.transformer_runner->session.Run(run_options, *state.transformer_io);
        auto velocity_ready = state.dit_velocity->CopyAsyncToHostWithNotification();
        velocity_ready.Sync();
    }
    const auto dit_done = std::chrono::steady_clock::now();

    {
        din::common::nvtx_scoped_range nvtx("cfg_data_prediction");
        // Velocity is already on the host.
        state.cfg.Run(compute_stream(state), SamplingBackend::Cpu, t, state.sampler.cfg_scale, x0);
    }
    const auto cfg_done = std::chrono::steady_clock::now();
    state.timing.upload += uploaded - start;
    state.timing.dit += dit_done - uploaded;
    state.timing.cfg += cfg_done - dit_done;
    ++state.timing.evaluations;
}

PixelDiTImage run_pipeline(PipelineState& state, const PixelDiTConfig& config, unsigned int seed)
{
    const PixelDiTPipelineConfig& pc = state.pipeline_config;
    const bool dump = !config.dump_dir.empty();
    const size_t n = element_count(*state.sample);
    if (!config.init_noise_path.empty())
    {
        const std::vector<float> noise = read_raw_floats(config.init_noise_path, n);
        std::copy(noise.begin(), noise.end(), state.sample->HostData());
        std::cout << "Initial noise loaded from " << config.init_noise_path.string() << std::endl;
    }
    else
    {
        initialize_noise(state.sample->HostData(), n, seed);
    }
    if (dump)
    {
        write_raw(config.dump_dir / ("noise_seed" + std::to_string(seed) + ".bin"), state.sample->HostData(), n);
    }

    const int steps = state.sampler.steps;
    const std::vector<float> timesteps = flow_timesteps(steps, state.sampler.flow_shift, pc.t_start, pc.t_end);
    if (dump)
    {
        write_raw(config.dump_dir / "timesteps.bin", timesteps.data(), timesteps.size());
    }

    const bool cuda = state.sampling_backend == SamplingBackend::Cuda;
    state.timing = {};
    const auto start = std::chrono::steady_clock::now();
    if (cuda)
    {
        // Model evaluations happen at timesteps[0 .. steps-1].
        float* table = state.timestep_table->HostData();
        const size_t batch = static_cast<size_t>(pc.dit_batch);
        for (size_t i = 0; i < static_cast<size_t>(steps); ++i)
        {
            std::fill_n(table + i * batch, batch, timesteps[i] * pc.timestep_scale);
        }
        auto sample_uploaded = state.sample->CopyAsyncToDeviceWithNotification();
        auto table_uploaded = state.timestep_table->CopyAsyncToDeviceWithNotification();
        sample_uploaded.Sync();
        table_uploaded.Sync();
    }

    // FlowDPMSolver.sample (model_export/pixeldit/pipeline.py): 1st order first, 2nd order in the middle,
    // 1st order last (lower_order_final), and no model evaluation after the final step.
    din::common::TensorBuffer<float>* prev = state.model_a.get();  // x0 at t_prev_1
    din::common::TensorBuffer<float>* curr = state.model_b.get();  // x0 at t_prev_0
    float t_prev_1 = 0.0f;
    float t_prev_0 = timesteps[0];
    data_prediction(state, timesteps[0], 0, *curr);
    int history = 1;
    for (int step = 1; step <= steps; ++step)
    {
        din::common::nvtx_scoped_range nvtx_step(("sampling_step_" + std::to_string(step)).c_str());
        const float t = timesteps[static_cast<size_t>(step)];
        const int step_order = step < pc.order ? 1 : std::min(pc.order, steps + 1 - step);
        if (step_order == 2 && history < 2)
        {
            throw std::logic_error("2nd-order update without two x0 estimates");
        }
        const auto solver_start = std::chrono::steady_clock::now();
        state.solver.Run(compute_stream(state), state.sampling_backend, step_order, prev, *curr, t_prev_1, t_prev_0,
                         t);
        state.timing.solver += std::chrono::steady_clock::now() - solver_start;
        std::swap(prev, curr);
        t_prev_1 = t_prev_0;
        t_prev_0 = t;
        history = std::min(history + 1, 2);
        if (step < steps)
        {
            data_prediction(state, t, step, *curr);
        }
        // With --processing cuda this reports enqueued steps; the GPU may still be behind.
        if (step % 10 == 0 || step == steps)
        {
            std::cout << "  step " << step << "/" << steps << std::endl;
        }
    }
    if (cuda)
    {
        auto sample_ready = state.sample->CopyAsyncToHostWithNotification();
        sample_ready.Sync();
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Sampling: " << elapsed.count() << " s (" << elapsed.count() / steps * 1000.0 << " ms/step)"
              << std::endl;
    if (!cuda)
    {
        const double evals = std::max(state.timing.evaluations, 1);
        std::cout << "  per DiT evaluation: upload " << state.timing.upload.count() / evals * 1000.0
                  << " ms, DiT + download " << state.timing.dit.count() / evals * 1000.0 << " ms, CFG "
                  << state.timing.cfg.count() / evals * 1000.0 << " ms; solver "
                  << state.timing.solver.count() / steps * 1000.0 << " ms/step" << std::endl;
    }

    if (dump)
    {
        write_raw(config.dump_dir / ("final_seed" + std::to_string(seed) + ".bin"), state.sample->HostData(), n);
    }
    PixelDiTImage image;
    image.height = static_cast<int>(pc.height);
    image.width = static_cast<int>(pc.width);
    image.data.assign(state.sample->HostData(), state.sample->HostData() + n);
    return image;
}

class TrtPixelDiTProcessingPipeline final : public PixelDiTProcessingPipeline
{
public:
    TrtPixelDiTProcessingPipeline(PixelDiTConfig config, PixelDiTRuntimeContext& runtime)
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
        if (std::strcmp(runtime_.trt_device.EpName(), kDinNvTensorRTRTXExecutionProvider) != 0)
        {
            throw std::runtime_error("TensorRT RTX execution provider requested but no device was found");
        }
        if (!config_.dump_dir.empty())
        {
            std::filesystem::create_directories(config_.dump_dir);
        }
        auto state = std::make_unique<PipelineState>(runtime_.env);
        state->sampling_backend =
            config_.processing == PixelDiTProcessingBackend::Cuda ? SamplingBackend::Cuda : SamplingBackend::Cpu;
        std::cout << "Sampling backend: " << to_string(state->sampling_backend) << std::endl;
        initialize_state(*state, runtime_.trt_device, config_);
        state->prompt = config_.prompt;
        state_ = std::move(state);
    }

    void SetPrompt(std::string prompt) override
    {
        config_.prompt = std::move(prompt);
        if (state_)
        {
            state_->prompt = config_.prompt;
            state_->prompt_embeds_valid = false;
        }
    }

    PixelDiTImage GenerateImage(unsigned int seed) override
    {
        if (!state_)
        {
            throw std::runtime_error("PixelDiT pipeline was not initialized");
        }
        if (!state_->prompt_embeds_valid)
        {
            encode_prompts(*state_, config_);
        }
        return run_pipeline(*state_, config_, seed);
    }

private:
    PixelDiTConfig config_;
    PixelDiTRuntimeContext& runtime_;
    std::unique_ptr<PipelineState> state_;
};

}  // namespace

std::unique_ptr<PixelDiTProcessingPipeline> CreatePixelDiTTrtPipeline(const PixelDiTConfig& config,
                                                                       PixelDiTRuntimeContext& runtime)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    return std::make_unique<TrtPixelDiTProcessingPipeline>(config, runtime);
}

}  // namespace din::image_gen
