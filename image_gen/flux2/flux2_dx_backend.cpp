// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dx_helper.h"
#include "flux2.h"
#include "flux2_runtime_context.h"
#include "ort_session.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>

class OrtGraphicsInteropScope
{
public:
    OrtGraphicsInteropScope(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device,
                            ID3D12CommandQueue* command_queue)
        : interop_(interop)
        , ep_device_(ep_device)
    {
        OrtGraphicsInteropConfig config{};
        config.version = ORT_API_VERSION;
        config.graphics_api = ORT_GRAPHICS_API_D3D12;
        config.command_queue = command_queue;
        config.additional_options = nullptr;

        Ort::ThrowOnError(interop_.InitGraphicsInteropForEpDevice(ep_device_, &config));
        active_ = true;
    }

    OrtGraphicsInteropScope(const OrtGraphicsInteropScope&) = delete;
    OrtGraphicsInteropScope& operator=(const OrtGraphicsInteropScope&) = delete;

    ~OrtGraphicsInteropScope()
    {
        if (!active_)
        {
            return;
        }

        OrtStatus* status = interop_.DeinitGraphicsInteropForEpDevice(ep_device_);
        if (status != nullptr)
        {
            const OrtApi& api = Ort::GetApi();
            std::cerr << "DeinitGraphicsInteropForEpDevice failed: " << api.GetErrorMessage(status) << std::endl;
            api.ReleaseStatus(status);
        }
    }

private:
    const OrtInteropApi& interop_;
    Ort::ConstEpDevice ep_device_;
    bool active_ = false;
};

class OrtD3D12Fence
{
public:
    OrtD3D12Fence(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device, ID3D12Device* device)
        : interop_(interop)
    {
        if (device == nullptr)
        {
            throw std::runtime_error("Cannot create ORT/D3D12 fence without a D3D12 device");
        }

        dx_check(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)), "CreateFence ORT/D3D12 sync");
        dx_check(device->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle_),
                 "CreateSharedHandle ORT/D3D12 sync fence");
        if (shared_handle_ == nullptr)
        {
            throw std::runtime_error("CreateSharedHandle returned a null fence handle");
        }

        Ort::ThrowOnError(interop_.CreateExternalResourceImporterForDevice(ep_device, &importer_));
        if (importer_ == nullptr)
        {
            throw std::runtime_error("CreateExternalResourceImporterForDevice returned null importer");
        }

        OrtExternalSemaphoreDescriptor desc{};
        desc.version = ORT_API_VERSION;
        desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
        desc.native_handle = shared_handle_;
        Ort::ThrowOnError(interop_.ImportSemaphore(importer_, &desc, &semaphore_));
        if (semaphore_ == nullptr)
        {
            throw std::runtime_error("ImportSemaphore returned null semaphore");
        }
    }

    OrtD3D12Fence(const OrtD3D12Fence&) = delete;
    OrtD3D12Fence& operator=(const OrtD3D12Fence&) = delete;

    ~OrtD3D12Fence()
    {
        if (semaphore_ != nullptr)
        {
            interop_.ReleaseExternalSemaphoreHandle(semaphore_);
        }
        if (importer_ != nullptr)
        {
            interop_.ReleaseExternalResourceImporter(importer_);
        }
        if (shared_handle_ != nullptr)
        {
            CloseHandle(shared_handle_);
        }
    }

    uint64_t signal_ort(Ort::SyncStream& stream)
    {
        const uint64_t value = next_value();
        Ort::ThrowOnError(interop_.SignalSemaphore(importer_, semaphore_, stream, value));
        return value;
    }

    void wait_ort(Ort::SyncStream& stream, uint64_t value)
    {
        Ort::ThrowOnError(interop_.WaitSemaphore(importer_, semaphore_, stream, value));
    }

    uint64_t signal_d3d12(ID3D12CommandQueue* queue)
    {
        const uint64_t value = next_value();
        dx_check(queue->Signal(fence_.Get(), value), "D3D12 Signal ORT/D3D12 sync fence");
        return value;
    }

    void wait_d3d12(ID3D12CommandQueue* queue, uint64_t value)
    {
        dx_check(queue->Wait(fence_.Get(), value), "D3D12 Wait ORT/D3D12 sync fence");
    }

private:
    uint64_t next_value()
    {
        return ++fence_value_;
    }

    const OrtInteropApi& interop_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE shared_handle_ = nullptr;
    OrtExternalResourceImporter* importer_ = nullptr;
    OrtExternalSemaphoreHandle* semaphore_ = nullptr;
    uint64_t fence_value_ = 0;
};

class OrtD3D12TensorImporter
{
public:
    OrtD3D12TensorImporter(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device, ID3D12Device* device)
        : interop_(interop)
        , device_(device)
    {
        if (device_ == nullptr)
        {
            throw std::runtime_error("Cannot import D3D12 tensors without a D3D12 device");
        }

        Ort::ThrowOnError(interop_.CreateExternalResourceImporterForDevice(ep_device, &importer_));
        if (importer_ == nullptr)
        {
            throw std::runtime_error("CreateExternalResourceImporterForDevice returned null importer");
        }

        bool can_import_resource = false;
        Ort::ThrowOnError(
            interop_.CanImportMemory(importer_, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, &can_import_resource));
        if (!can_import_resource)
        {
            throw std::runtime_error("ORT external resource importer cannot import D3D12 resources");
        }
    }

    OrtD3D12TensorImporter(const OrtD3D12TensorImporter&) = delete;
    OrtD3D12TensorImporter& operator=(const OrtD3D12TensorImporter&) = delete;

    ~OrtD3D12TensorImporter()
    {
        for (ImportedMemory& memory : imported_memory_)
        {
            if (memory.memory != nullptr)
            {
                interop_.ReleaseExternalMemoryHandle(memory.memory);
            }
            if (memory.shared_handle != nullptr)
            {
                CloseHandle(memory.shared_handle);
            }
        }
        if (importer_ != nullptr)
        {
            interop_.ReleaseExternalResourceImporter(importer_);
        }
    }

    Ort::Value create_tensor(DxBuffer& buffer, const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
    {
        if (buffer.resource == nullptr)
        {
            throw std::runtime_error("Cannot import a null D3D12 resource as an ORT tensor");
        }

        HANDLE shared_handle = nullptr;
        dx_check(device_->CreateSharedHandle(buffer.resource.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle),
                 "CreateSharedHandle ORT/D3D12 tensor resource");
        if (shared_handle == nullptr)
        {
            throw std::runtime_error("CreateSharedHandle returned a null tensor resource handle");
        }

        OrtExternalMemoryDescriptor memory_desc{};
        memory_desc.version = ORT_API_VERSION;
        memory_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        memory_desc.native_handle = shared_handle;
        memory_desc.size_bytes = buffer.size;
        memory_desc.offset_bytes = 0;

        OrtExternalMemoryHandle* memory = nullptr;
        OrtStatus* status = interop_.ImportMemory(importer_, &memory_desc, &memory);
        if (status != nullptr)
        {
            CloseHandle(shared_handle);
            Ort::ThrowOnError(status);
        }
        if (memory == nullptr)
        {
            CloseHandle(shared_handle);
            throw std::runtime_error("ImportMemory returned null memory handle");
        }

        OrtExternalTensorDescriptor tensor_desc{};
        tensor_desc.version = ORT_API_VERSION;
        tensor_desc.element_type = type;
        tensor_desc.shape = shape.data();
        tensor_desc.rank = shape.size();
        tensor_desc.offset_bytes = 0;

        OrtValue* tensor = nullptr;
        status = interop_.CreateTensorFromMemory(importer_, memory, &tensor_desc, &tensor);
        if (status != nullptr)
        {
            interop_.ReleaseExternalMemoryHandle(memory);
            CloseHandle(shared_handle);
            Ort::ThrowOnError(status);
        }
        if (tensor == nullptr)
        {
            interop_.ReleaseExternalMemoryHandle(memory);
            CloseHandle(shared_handle);
            throw std::runtime_error("CreateTensorFromMemory returned null tensor");
        }

        imported_memory_.push_back({shared_handle, memory});
        return Ort::Value(tensor);
    }

private:
    struct ImportedMemory
    {
        HANDLE shared_handle = nullptr;
        OrtExternalMemoryHandle* memory = nullptr;
    };

    const OrtInteropApi& interop_;
    ID3D12Device* device_;
    OrtExternalResourceImporter* importer_ = nullptr;
    std::vector<ImportedMemory> imported_memory_;
};

static void run_pipeline_dx(Ort::Session& text_encoder_session, Ort::Session& transformer_session,
                            Ort::Session& vae_decoder_session, Ort::IoBinding& text_encoder_io,
                            Ort::IoBinding& transformer_io, Ort::IoBinding& vae_decoder_io,
                            const std::vector<float>& time_schedule, DxContext& dx, DxEulerPipeline& euler_pipeline,
                            DxPostprocessPipeline& postprocess_pipeline, OrtD3D12Fence& sync_fence,
                            Ort::SyncStream& ort_stream, DxBuffer& timestep_buf, DxBuffer& hidden_states,
                            DxBuffer& transformer_output, DxBuffer& decoder_latent, DxBuffer& bn_mean, DxBuffer& bn_std,
                            bool encode_prompt)
{
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");

    if (encode_prompt)
    {
        text_encoder_session.Run(run_options, text_encoder_io);
    }

    dx.begin();
    dx.record_upload(timestep_buf, &time_schedule[0], sizeof(float));
    dx.execute();
    const uint64_t initial_timestep_uploaded = sync_fence.signal_d3d12(dx.queue.Get());
    sync_fence.wait_ort(ort_stream, initial_timestep_uploaded);

    for (int step = 0; step < FLOW_STEPS; ++step)
    {
        const float t_curr = time_schedule[step];
        const float t_next = time_schedule[step + 1];

        transformer_session.Run(run_options, transformer_io);

        const uint64_t transformer_done = sync_fence.signal_ort(ort_stream);
        sync_fence.wait_d3d12(dx.queue.Get(), transformer_done);

        DxEulerConstants euler{};
        euler.t_curr = t_curr;
        euler.t_next = t_next;
        euler.total_elements = static_cast<uint32_t>(hidden_states.size / sizeof(float));

        dx.begin();
        euler_pipeline.record(dx, hidden_states, transformer_output, euler);
        if (step + 1 < FLOW_STEPS)
        {
            dx.record_upload(timestep_buf, &t_next, sizeof(float));
        }
        dx.execute();

        const uint64_t d3d_done = sync_fence.signal_d3d12(dx.queue.Get());
        if (step + 1 < FLOW_STEPS)
        {
            sync_fence.wait_ort(ort_stream, d3d_done);
        }
    }

    DxPostprocessConstants constants{};
    constants.C = static_cast<uint32_t>(LATENT_CHANNELS);
    constants.I = static_cast<uint32_t>(LATENT_HEIGHT);
    constants.J = static_cast<uint32_t>(LATENT_WIDTH);
    constants.pi = static_cast<uint32_t>(PATCH_SIZE);
    constants.pj = static_cast<uint32_t>(PATCH_SIZE);
    constants.total_elements = static_cast<uint32_t>((LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE) *
                                                     (LATENT_HEIGHT * PATCH_SIZE) * (LATENT_WIDTH * PATCH_SIZE));

    postprocess_pipeline.dispatch(dx, hidden_states, decoder_latent, bn_mean, bn_std, constants);
    const uint64_t postprocess_done = sync_fence.signal_d3d12(dx.queue.Get());
    sync_fence.wait_ort(ort_stream, postprocess_done);

    vae_decoder_session.Run(run_options, vae_decoder_io);
    const uint64_t vae_done = sync_fence.signal_ort(ort_stream);
    sync_fence.wait_d3d12(dx.queue.Get(), vae_done);
}

struct DxPipelineState
{
    DxPipelineState(Ort::Env& environment, uint64_t ort_luid)
        : dx(ort_luid)
        , env(environment)
    {
    }

    ~DxPipelineState()
    {
        text_encoder_io.reset();
        transformer_io.reset();
        vae_decoder_io.reset();

        token_tensor = Ort::Value{nullptr};
        attention_mask_tensor = Ort::Value{nullptr};
        text_encoder_embeds_tensor = Ort::Value{nullptr};
        timestep_tensor = Ort::Value{nullptr};
        hidden_states_tensor = Ort::Value{nullptr};
        img_ids_tensor = Ort::Value{nullptr};
        txt_ids_tensor = Ort::Value{nullptr};
        transformer_output_tensor = Ort::Value{nullptr};
        decoder_input_tensor = Ort::Value{nullptr};
        decoded_image_tensor = Ort::Value{nullptr};

        text_encoder_runner.reset();
        transformer_runner.reset();
        vae_decoder_runner.reset();

        tensor_importer.reset();
        sync_fence.reset();
        sync_stream.reset();
        graphics_interop.reset();

        euler_pipeline.reset();
        postprocess_pipeline.reset();
    }

    bool use_cig = false;
    bool prompt_embeds_valid = false;
    DxContext dx;
    Ort::Env& env;
    Ort::ConstEpDevice trt_device{};
    std::optional<Ort::SyncStream> sync_stream;
    const OrtInteropApi* interop_api = nullptr;

    std::unique_ptr<OrtGraphicsInteropScope> graphics_interop;
    std::unique_ptr<OrtD3D12Fence> sync_fence;
    std::unique_ptr<din::common::OrtRunner> text_encoder_runner;
    std::unique_ptr<din::common::OrtRunner> transformer_runner;
    std::unique_ptr<din::common::OrtRunner> vae_decoder_runner;

    std::unique_ptr<DxEulerPipeline> euler_pipeline;
    std::unique_ptr<DxPostprocessPipeline> postprocess_pipeline;
    std::unique_ptr<OrtD3D12TensorImporter> tensor_importer;

    DxBuffer token_buf;
    DxBuffer attn_mask_buf;
    DxBuffer text_encoder_embeds;
    DxBuffer timestep_buf;
    DxBuffer hidden_states;
    DxBuffer img_ids_buf;
    DxBuffer txt_ids_buf;
    DxBuffer transformer_output;
    DxBuffer decoder_latent;
    DxBuffer decoded_image_buf;
    DxBuffer bn_mean_buf;
    DxBuffer bn_std_buf;
    DxBuffer readback_buffer;
    size_t decoded_image_bytes = 0;
    size_t hidden_count = 0;

    Ort::Value token_tensor{nullptr};
    Ort::Value attention_mask_tensor{nullptr};
    Ort::Value text_encoder_embeds_tensor{nullptr};
    Ort::Value timestep_tensor{nullptr};
    Ort::Value hidden_states_tensor{nullptr};
    Ort::Value img_ids_tensor{nullptr};
    Ort::Value txt_ids_tensor{nullptr};
    Ort::Value transformer_output_tensor{nullptr};
    Ort::Value decoder_input_tensor{nullptr};
    Ort::Value decoded_image_tensor{nullptr};

    std::unique_ptr<Ort::IoBinding> text_encoder_io;
    std::unique_ptr<Ort::IoBinding> transformer_io;
    std::unique_ptr<Ort::IoBinding> vae_decoder_io;
    std::vector<float> time_schedule;
};

static void initialize_dx_state(DxPipelineState& state, const Flux2Config& config, Ort::ConstEpDevice trt_device)
{
    state.use_cig = config.processing == Flux2ProcessingBackend::DxCig;
    if (config.provider != Flux2ExecutionProvider::TrtRtx)
    {
        throw std::runtime_error("DirectX processing requires --provider trt-rtx");
    }
    const Flux2ModelPaths model_paths = MakeFlux2ModelPaths(config.model_dir, config.precision);
    const Flux2ModelCachePaths cache_paths =
        MakeFlux2ModelCachePaths(config.precision, state.use_cig ? "dx_cig" : "dx");

    std::cout << "Model dir: " << model_paths.base_dir.string() << "\n"
              << "CIG:    " << (state.use_cig ? "enabled" : "disabled") << "\n"
              << std::endl;

    std::cout << "\n=== Initializing D3D12 ===" << std::endl;
    state.trt_device = trt_device;
    state.interop_api = &Ort::GetInteropApi();

    state.graphics_interop = std::make_unique<OrtGraphicsInteropScope>(*state.interop_api, state.trt_device,
                                                                       state.use_cig ? state.dx.queue.Get() : nullptr);
    state.sync_fence = std::make_unique<OrtD3D12Fence>(*state.interop_api, state.trt_device, state.dx.device.Get());
    // A stream requires an active context - due to that interop needs to be initialized first
    state.sync_stream.emplace(din::common::CreateTensorRTRTXComputeStream(state.env));

    std::vector<std::pair<std::string, std::string>> graphics_ep_options;
    if (state.use_cig)
    {
        const int cuda_device_ordinal = din::common::ChooseCudaDeviceOrdinal(state.trt_device);
        const auto shared_memory_info =
            din::common::QueryCudaGraphicsInteropSharedMemoryInfo(cuda_device_ordinal, false);
        if (!shared_memory_info.supports_simultaneous_graphics_compute)
        {
            throw std::runtime_error(std::string("DirectX CIG is not supported on CUDA device generation ") +
                                     din::common::ToString(shared_memory_info.generation));
        }
        std::cout << "CUDA device ordinal " << shared_memory_info.cuda_device_ordinal << " compute capability "
                  << shared_memory_info.compute_capability_major << "." << shared_memory_info.compute_capability_minor
                  << " (" << din::common::ToString(shared_memory_info.generation) << ")"
                  << " graphics interop shared memory limit: " << (shared_memory_info.max_shared_memory_bytes / 1024)
                  << " KiB" << std::endl;
        graphics_ep_options.emplace_back("nv_max_shared_mem_size",
                                         std::to_string(shared_memory_info.max_shared_memory_bytes));
        graphics_ep_options.emplace_back("nv_length_aux_stream_array", "0");
    }

    std::cout << "=== Loading ONNX Models ===" << std::endl;
    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config.ep_context_dir.string();
    const std::string cache_dir = config.ep_cache_dir.string();
    auto make_profile = [&](std::string cache_subpath)
    {
        din::common::ModelProfile profile;
        profile.cache_subpath = std::move(cache_subpath);
        profile.enable_cuda_graph = state.use_cig ? false : true;
        profile.embed_ep_context = false;
        profile.extra_ep_options = graphics_ep_options;
        return profile;
    };

    state.text_encoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.text_encoder_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.text_encoder),
        &*state.sync_stream);
    state.transformer_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.transformer_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.transformer),
        &*state.sync_stream);
    state.vae_decoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.vae_decoder_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.vae_decoder),
        &*state.sync_stream);

    std::vector<int64_t> token_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    std::vector<int64_t> attn_mask_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    std::vector<int64_t> te_out_shape = {BATCH_SIZE, SEQUENCE_LENGTH, TEXT_ENCODER_EMBED_DIM};
    std::vector<int64_t> timestep_shape = {BATCH_SIZE};
    std::vector<int64_t> hidden_shape = {BATCH_SIZE, TRANSFORMER_HIDDEN_DIM, LATENT_CHANNELS};
    std::vector<int64_t> img_ids_shape = {BATCH_SIZE, IMAGE_SEQUENCE, 4};
    std::vector<int64_t> txt_ids_shape = {BATCH_SIZE, SEQUENCE_LENGTH, 4};
    std::vector<int64_t> dec_latent_shape = {BATCH_SIZE, LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE,
                                             LATENT_HEIGHT * PATCH_SIZE, LATENT_WIDTH * PATCH_SIZE};
    std::vector<int64_t> image_shape = {BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH};

    state.token_buf = state.dx.create_shared_default_buffer(shape_numel(token_shape) * sizeof(int64_t));
    state.attn_mask_buf = state.dx.create_shared_default_buffer(shape_numel(attn_mask_shape) * sizeof(int64_t));
    state.text_encoder_embeds = state.dx.create_shared_default_buffer(shape_numel(te_out_shape) * sizeof(float));
    state.timestep_buf = state.dx.create_shared_default_buffer(sizeof(float));
    state.hidden_states = state.dx.create_shared_default_buffer(shape_numel(hidden_shape) * sizeof(float));
    state.img_ids_buf = state.dx.create_shared_default_buffer(shape_numel(img_ids_shape) * sizeof(int64_t));
    state.txt_ids_buf = state.dx.create_shared_default_buffer(shape_numel(txt_ids_shape) * sizeof(int64_t));
    state.transformer_output = state.dx.create_shared_default_buffer(shape_numel(hidden_shape) * sizeof(float));
    state.decoder_latent = state.dx.create_shared_default_buffer(shape_numel(dec_latent_shape) * sizeof(float));
    state.decoded_image_bytes = shape_numel(image_shape) * sizeof(float);
    state.hidden_count = shape_numel(hidden_shape);
    state.decoded_image_buf = state.dx.create_shared_default_buffer(state.decoded_image_bytes);
    state.bn_mean_buf = state.dx.create_default_buffer(LATENT_CHANNELS * sizeof(float));
    state.bn_std_buf = state.dx.create_default_buffer(LATENT_CHANNELS * sizeof(float));

    size_t upload_size = std::max(state.hidden_states.size, state.decoded_image_bytes);
    upload_size = std::max(upload_size, std::max(state.token_buf.size, state.img_ids_buf.size));
    state.dx.create_upload_ring(upload_size);
    state.readback_buffer = state.dx.create_readback_buffer(state.decoded_image_bytes);

    state.euler_pipeline = std::make_unique<DxEulerPipeline>(state.dx.device.Get());
    state.postprocess_pipeline = std::make_unique<DxPostprocessPipeline>(state.dx.device.Get());

    state.tensor_importer =
        std::make_unique<OrtD3D12TensorImporter>(*state.interop_api, state.trt_device, state.dx.device.Get());
    state.token_tensor =
        state.tensor_importer->create_tensor(state.token_buf, token_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    state.attention_mask_tensor =
        state.tensor_importer->create_tensor(state.attn_mask_buf, attn_mask_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    state.text_encoder_embeds_tensor = state.tensor_importer->create_tensor(state.text_encoder_embeds, te_out_shape,
                                                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    state.timestep_tensor =
        state.tensor_importer->create_tensor(state.timestep_buf, timestep_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    state.hidden_states_tensor =
        state.tensor_importer->create_tensor(state.hidden_states, hidden_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    state.img_ids_tensor =
        state.tensor_importer->create_tensor(state.img_ids_buf, img_ids_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    state.txt_ids_tensor =
        state.tensor_importer->create_tensor(state.txt_ids_buf, txt_ids_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    state.transformer_output_tensor = state.tensor_importer->create_tensor(state.transformer_output, hidden_shape,
                                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    state.decoder_input_tensor = state.tensor_importer->create_tensor(state.decoder_latent, dec_latent_shape,
                                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    state.decoded_image_tensor =
        state.tensor_importer->create_tensor(state.decoded_image_buf, image_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    state.text_encoder_io = std::make_unique<Ort::IoBinding>(state.text_encoder_runner->session);
    state.text_encoder_io->BindInput("input_ids", state.token_tensor);
    state.text_encoder_io->BindInput("attention_mask", state.attention_mask_tensor);
    state.text_encoder_io->BindOutput("prompt_embeds", state.text_encoder_embeds_tensor);

    state.transformer_io = std::make_unique<Ort::IoBinding>(state.transformer_runner->session);
    state.transformer_io->BindInput("hidden_states", state.hidden_states_tensor);
    state.transformer_io->BindInput("encoder_hidden_states", state.text_encoder_embeds_tensor);
    state.transformer_io->BindInput("timestep", state.timestep_tensor);
    state.transformer_io->BindInput("img_ids", state.img_ids_tensor);
    state.transformer_io->BindInput("txt_ids", state.txt_ids_tensor);
    state.transformer_io->BindOutput("sample", state.transformer_output_tensor);

    state.vae_decoder_io = std::make_unique<Ort::IoBinding>(state.vae_decoder_runner->session);
    state.vae_decoder_io->BindInput("latent_sample", state.decoder_input_tensor);
    state.vae_decoder_io->BindOutput("sample", state.decoded_image_tensor);

    std::cout << "\n=== Uploading Static Inputs (D3D12) ===" << std::endl;
    state.dx.begin();
    state.dx.record_upload(state.bn_mean_buf, BN_MEAN, state.bn_mean_buf.size);
    state.dx.record_upload(state.bn_std_buf, BN_STD, state.bn_std_buf.size);

    {
        std::vector<int64_t> img_ids_cpu(shape_numel(img_ids_shape));
        for (int64_t b = 0; b < BATCH_SIZE; ++b)
        {
            int64_t idx = 0;
            for (int64_t h = 0; h < LATENT_HEIGHT; ++h)
            {
                for (int64_t w = 0; w < LATENT_WIDTH; ++w, ++idx)
                {
                    int64_t base = (b * IMAGE_SEQUENCE + idx) * 4;
                    img_ids_cpu[base + 0] = 0;
                    img_ids_cpu[base + 1] = h;
                    img_ids_cpu[base + 2] = w;
                    img_ids_cpu[base + 3] = 0;
                }
            }
        }
        state.dx.record_upload(state.img_ids_buf, img_ids_cpu.data(), state.img_ids_buf.size);
    }

    {
        std::vector<int64_t> txt_ids_cpu(shape_numel(txt_ids_shape));
        for (int64_t b = 0; b < BATCH_SIZE; ++b)
        {
            for (int64_t t = 0; t < SEQUENCE_LENGTH; ++t)
            {
                int64_t base = (b * SEQUENCE_LENGTH + t) * 4;
                txt_ids_cpu[base + 0] = 0;
                txt_ids_cpu[base + 1] = 0;
                txt_ids_cpu[base + 2] = 0;
                txt_ids_cpu[base + 3] = t;
            }
        }
        state.dx.record_upload(state.txt_ids_buf, txt_ids_cpu.data(), state.txt_ids_buf.size);
    }

    constexpr double MU = 2.291179894115571;
    state.time_schedule.resize(FLOW_STEPS + 1);
    for (int i = 0; i <= FLOW_STEPS; ++i)
    {
        double t = T_START - (T_START - T_END) * static_cast<double>(i) / FLOW_STEPS;
        state.time_schedule[i] = static_cast<float>(std::exp(MU) / (std::exp(MU) + std::pow(1.0 / t - 1.0, 1.0)));
        std::cout << "  sigma[" << i << "] = " << state.time_schedule[i] << std::endl;
    }

    std::cout << "\n=== Tokenizing Prompt ===" << std::endl;
    {
        const Flux2TextEncoderInputs text_inputs = TokenizeFlux2Prompt(model_paths, config.prompt);

        std::vector<int64_t> tokens_cpu(shape_numel(token_shape));
        std::vector<int64_t> attn_mask_cpu(shape_numel(attn_mask_shape));
        FillTextEncoderInputs(text_inputs.token_ids, text_inputs.pad_token_id, tokens_cpu.data(), attn_mask_cpu.data(),
                              BATCH_SIZE, SEQUENCE_LENGTH);

        state.dx.record_upload(state.token_buf, tokens_cpu.data(), state.token_buf.size);
        state.dx.record_upload(state.attn_mask_buf, attn_mask_cpu.data(), state.attn_mask_buf.size);
    }
    state.dx.execute();
    const uint64_t static_uploads_done = state.sync_fence->signal_d3d12(state.dx.queue.Get());
    state.sync_fence->wait_ort(*state.sync_stream, static_uploads_done);
}

static Flux2Image run_dx_image(DxPipelineState& state, unsigned int seed)
{
    std::vector<float> latent_cpu(state.hidden_count);
    initialize_latent(latent_cpu.data(), latent_cpu.size(), seed);
    state.dx.begin();
    state.dx.record_upload(state.hidden_states, latent_cpu.data(), state.hidden_states.size);
    state.dx.execute();
    const uint64_t latent_uploaded = state.sync_fence->signal_d3d12(state.dx.queue.Get());
    state.sync_fence->wait_ort(*state.sync_stream, latent_uploaded);

    run_pipeline_dx(state.text_encoder_runner->session, state.transformer_runner->session,
                    state.vae_decoder_runner->session, *state.text_encoder_io, *state.transformer_io,
                    *state.vae_decoder_io, state.time_schedule, state.dx, *state.euler_pipeline,
                    *state.postprocess_pipeline, *state.sync_fence, *state.sync_stream, state.timestep_buf,
                    state.hidden_states, state.transformer_output, state.decoder_latent, state.bn_mean_buf,
                    state.bn_std_buf, !state.prompt_embeds_valid);
    state.prompt_embeds_valid = true;

    state.dx.readback(state.decoded_image_buf, state.readback_buffer, state.decoded_image_bytes);
    void* mapped = state.dx.map(state.readback_buffer);
    const float* image_data = static_cast<const float*>(mapped);
    Flux2Image image;
    image.height = static_cast<int>(IMAGE_HEIGHT);
    image.width = static_cast<int>(IMAGE_WIDTH);
    image.data.resize(state.decoded_image_bytes / sizeof(float));
    std::copy_n(image_data, image.data.size(), image.data.data());
    state.dx.unmap(state.readback_buffer);

    return image;
}

class DxFlux2ProcessingPipeline final : public Flux2ProcessingPipeline
{
public:
    DxFlux2ProcessingPipeline(Flux2Config config, Flux2RuntimeContext& runtime)
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
        std::cout << "Initializing Flux2 DirectX pipeline" << std::endl;
        const auto identity = din::common::ResolveOrtGraphicsDeviceIdentity(runtime_.trt_device);
        state_ = std::make_unique<DxPipelineState>(runtime_.env, identity.luid);
        initialize_dx_state(*state_, config_, runtime_.trt_device);
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
        std::vector<int64_t> tokens_cpu(BATCH_SIZE * SEQUENCE_LENGTH);
        std::vector<int64_t> attn_mask_cpu(BATCH_SIZE * SEQUENCE_LENGTH);
        FillTextEncoderInputs(text_inputs.token_ids, text_inputs.pad_token_id, tokens_cpu.data(), attn_mask_cpu.data(),
                              BATCH_SIZE, SEQUENCE_LENGTH);

        state_->dx.begin();
        state_->dx.record_upload(state_->token_buf, tokens_cpu.data(), state_->token_buf.size);
        state_->dx.record_upload(state_->attn_mask_buf, attn_mask_cpu.data(), state_->attn_mask_buf.size);
        state_->dx.execute();
        const uint64_t prompt_uploaded = state_->sync_fence->signal_d3d12(state_->dx.queue.Get());
        state_->sync_fence->wait_ort(*state_->sync_stream, prompt_uploaded);
    }

    Flux2Image GenerateImage(unsigned int seed) override
    {
        if (!state_)
        {
            throw std::runtime_error("Flux2 DirectX pipeline was not initialized");
        }
        return run_dx_image(*state_, seed);
    }

private:
    Flux2Config config_;
    Flux2RuntimeContext& runtime_;
    std::unique_ptr<DxPipelineState> state_;
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2DxPipeline(const Flux2Config& config, Flux2RuntimeContext& runtime)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    return std::make_unique<DxFlux2ProcessingPipeline>(config, runtime);
}
