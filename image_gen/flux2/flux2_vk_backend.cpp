// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nvtx3/nvtx3.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "flux2.h"
#include "flux2_runtime_context.h"
#include "ort_session.h"
#include "utils.h"
#include "vk_helper.h"

static const std::string EULER_SHADER_PATH = std::string(SHADER_DIR) + "euler.spv";
static const std::string POSTPROCESS_SHADER_PATH = std::string(SHADER_DIR) + "postprocess.spv";

// ============================================================================
// Upload latent noise for a new seed  (CPU → staging → GPU via Vulkan copy)
// ============================================================================

void initialize_latents(unsigned int seed, size_t hidden_count, void* dest)
{
    nvtx3::scoped_range nvtx("initialize_latents");
    std::cout << "\n=== Initializing Latent (seed=" << seed << ") ===" << std::endl;

    initialize_latent(static_cast<float*>(dest), hidden_count, seed);
}

class OrtVulkanGraphicsInteropScope
{
public:
    OrtVulkanGraphicsInteropScope(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device,
                                  const std::vector<uint8_t>& external_compute_queue_data)
        : interop_(interop)
        , ep_device_(ep_device)
    {
        if (external_compute_queue_data.empty())
        {
            throw std::runtime_error("Vulkan CIG external compute queue data is empty");
        }

        Ort::KeyValuePairs options;
        options.Add("VkExternalComputeQueueDataParamsNV_data",
                    std::to_string(reinterpret_cast<uintptr_t>(external_compute_queue_data.data())).c_str());

        OrtGraphicsInteropConfig config{};
        config.version = ORT_API_VERSION;
        config.graphics_api = ORT_GRAPHICS_API_VULKAN;
        config.command_queue = nullptr;
        config.additional_options = options.GetConst();
        Ort::ThrowOnError(interop_.InitGraphicsInteropForEpDevice(ep_device_, &config));
        active_ = true;
    }

    OrtVulkanGraphicsInteropScope(const OrtVulkanGraphicsInteropScope&) = delete;
    OrtVulkanGraphicsInteropScope& operator=(const OrtVulkanGraphicsInteropScope&) = delete;

    ~OrtVulkanGraphicsInteropScope()
    {
        if (!active_)
        {
            return;
        }
        OrtStatus* status = interop_.DeinitGraphicsInteropForEpDevice(ep_device_);
        if (status != nullptr)
        {
            std::cerr << "DeinitGraphicsInteropForEpDevice failed: " << Ort::GetApi().GetErrorMessage(status)
                      << std::endl;
            Ort::GetApi().ReleaseStatus(status);
        }
    }

private:
    const OrtInteropApi& interop_;
    Ort::ConstEpDevice ep_device_;
    bool active_ = false;
};

class OrtVulkanTensorImporter
{
public:
    OrtVulkanTensorImporter(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device, const VkHelper::Device& device)
        : interop_(interop)
        , device_(device)
    {
        Ort::ThrowOnError(interop_.CreateExternalResourceImporterForDevice(ep_device, &importer_));
        if (importer_ == nullptr)
        {
            throw std::runtime_error("CreateExternalResourceImporterForDevice returned null Vulkan importer");
        }

        bool can_import_memory = false;
        Ort::ThrowOnError(interop_.CanImportMemory(importer_, kOrtExternalMemoryHandleType, &can_import_memory));
        if (!can_import_memory)
        {
            throw std::runtime_error("ORT external resource importer cannot import Vulkan memory");
        }
    }

    OrtVulkanTensorImporter(const OrtVulkanTensorImporter&) = delete;
    OrtVulkanTensorImporter& operator=(const OrtVulkanTensorImporter&) = delete;

    ~OrtVulkanTensorImporter()
    {
        for (ImportedMemory& memory : imported_memory_)
        {
            if (memory.memory != nullptr)
            {
                interop_.ReleaseExternalMemoryHandle(memory.memory);
            }
            close_native_handle(memory.native_handle);
        }
        if (importer_ != nullptr)
        {
            interop_.ReleaseExternalResourceImporter(importer_);
        }
    }

    Ort::Value create_tensor(VulkanBuffer& buffer, const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
    {
        NativeHandle native_handle = device_.getMemoryHandle(buffer.memory);
        if (!is_valid_native_handle(native_handle))
        {
            throw std::runtime_error("Failed to export Vulkan memory handle for ORT tensor");
        }

        OrtExternalMemoryDescriptor memory_desc{};
        memory_desc.version = ORT_API_VERSION;
        memory_desc.handle_type = kOrtExternalMemoryHandleType;
        memory_desc.native_handle = to_ort_native_handle(native_handle);
        memory_desc.size_bytes = buffer.size;
        memory_desc.offset_bytes = 0;

        OrtExternalMemoryHandle* memory = nullptr;
        OrtStatus* status = interop_.ImportMemory(importer_, &memory_desc, &memory);
        if (status != nullptr)
        {
            close_native_handle(native_handle);
            Ort::ThrowOnError(status);
        }
        if (memory == nullptr)
        {
            close_native_handle(native_handle);
            throw std::runtime_error("ImportMemory returned null Vulkan memory handle");
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
            close_native_handle(native_handle);
            Ort::ThrowOnError(status);
        }
        if (tensor == nullptr)
        {
            interop_.ReleaseExternalMemoryHandle(memory);
            close_native_handle(native_handle);
            throw std::runtime_error("CreateTensorFromMemory returned null Vulkan tensor");
        }

        imported_memory_.push_back({native_handle, memory});
        return Ort::Value(tensor);
    }

private:
#ifdef _WIN32
    using NativeHandle = HANDLE;
    static constexpr OrtExternalMemoryHandleType kOrtExternalMemoryHandleType =
        ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32;

    static bool is_valid_native_handle(NativeHandle handle)
    {
        return handle != nullptr;
    }
    static void* to_ort_native_handle(NativeHandle handle)
    {
        return handle;
    }
    static void close_native_handle(NativeHandle handle)
    {
        if (handle != nullptr)
        {
            CloseHandle(handle);
        }
    }
#else
    using NativeHandle = int;
    static constexpr OrtExternalMemoryHandleType kOrtExternalMemoryHandleType =
        ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD;

    static bool is_valid_native_handle(NativeHandle handle)
    {
        return handle >= 0;
    }
    static void* to_ort_native_handle(NativeHandle handle)
    {
        return reinterpret_cast<void*>(static_cast<intptr_t>(handle));
    }
    static void close_native_handle(NativeHandle handle)
    {
        if (handle >= 0)
        {
            close(handle);
        }
    }
#endif

    struct ImportedMemory
    {
        NativeHandle native_handle{};
        OrtExternalMemoryHandle* memory = nullptr;
    };

    const OrtInteropApi& interop_;
    const VkHelper::Device& device_;
    OrtExternalResourceImporter* importer_ = nullptr;
    std::vector<ImportedMemory> imported_memory_;
};

class OrtVulkanTimelineSemaphore
{
public:
    OrtVulkanTimelineSemaphore(const OrtInteropApi& interop, Ort::ConstEpDevice ep_device, VkHelper::Device& device)
        : interop_(interop)
        , device_(device)
        , semaphore_(device_.createTimelineSemaphore(0, true))
    {
        Ort::ThrowOnError(interop_.CreateExternalResourceImporterForDevice(ep_device, &importer_));
        if (importer_ == nullptr)
        {
            throw std::runtime_error("CreateExternalResourceImporterForDevice returned null Vulkan semaphore importer");
        }

        bool can_import_semaphore = false;
        Ort::ThrowOnError(interop_.CanImportSemaphore(importer_, kOrtExternalSemaphoreType, &can_import_semaphore));
        if (!can_import_semaphore)
        {
            throw std::runtime_error("ORT external resource importer cannot import Vulkan timeline semaphores");
        }

        native_handle_ = device_.getSemaphoreHandle(semaphore_);
        if (!is_valid_native_handle(native_handle_))
        {
            throw std::runtime_error("Failed to export Vulkan timeline semaphore handle");
        }

        OrtExternalSemaphoreDescriptor desc{};
        desc.version = ORT_API_VERSION;
        desc.type = kOrtExternalSemaphoreType;
        desc.native_handle = to_ort_native_handle(native_handle_);
        Ort::ThrowOnError(interop_.ImportSemaphore(importer_, &desc, &ort_semaphore_));
        if (ort_semaphore_ == nullptr)
        {
            throw std::runtime_error("ImportSemaphore returned null Vulkan timeline semaphore handle");
        }
    }

    OrtVulkanTimelineSemaphore(const OrtVulkanTimelineSemaphore&) = delete;
    OrtVulkanTimelineSemaphore& operator=(const OrtVulkanTimelineSemaphore&) = delete;

    ~OrtVulkanTimelineSemaphore()
    {
        if (ort_semaphore_ != nullptr)
        {
            interop_.ReleaseExternalSemaphoreHandle(ort_semaphore_);
        }
        if (importer_ != nullptr)
        {
            interop_.ReleaseExternalResourceImporter(importer_);
        }
        close_native_handle(native_handle_);
    }

    uint64_t signal_ort(Ort::SyncStream& stream)
    {
        const uint64_t value = next_value();
        Ort::ThrowOnError(interop_.SignalSemaphore(importer_, ort_semaphore_, stream, value));
        return value;
    }

    void wait_ort(Ort::SyncStream& stream, uint64_t value)
    {
        Ort::ThrowOnError(interop_.WaitSemaphore(importer_, ort_semaphore_, stream, value));
    }

    uint64_t submit_vulkan(VkHelper& vk, VkCommandBuffer cmd, uint64_t wait_value = 0)
    {
        const uint64_t signal_value = next_value();
        vk.submitVulkanAsync(cmd, semaphore_, wait_value, signal_value);
        return signal_value;
    }

    void wait_vulkan(const VkHelper::Device& device, uint64_t value)
    {
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &semaphore_;
        wait_info.pValues = &value;
        VK_CHECK(vkWaitSemaphores(device.raw(), &wait_info, UINT64_MAX));
    }

private:
#ifdef _WIN32
    using NativeHandle = HANDLE;
    static constexpr OrtExternalSemaphoreType kOrtExternalSemaphoreType =
        ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;

    static bool is_valid_native_handle(NativeHandle handle)
    {
        return handle != nullptr;
    }
    static void* to_ort_native_handle(NativeHandle handle)
    {
        return handle;
    }
    static void close_native_handle(NativeHandle handle)
    {
        if (handle != nullptr)
        {
            CloseHandle(handle);
        }
    }
#else
    using NativeHandle = int;
    static constexpr OrtExternalSemaphoreType kOrtExternalSemaphoreType =
        ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;

    static bool is_valid_native_handle(NativeHandle handle)
    {
        return handle >= 0;
    }
    static void* to_ort_native_handle(NativeHandle handle)
    {
        return reinterpret_cast<void*>(static_cast<intptr_t>(handle));
    }
    static void close_native_handle(NativeHandle handle)
    {
        if (handle >= 0)
        {
            close(handle);
        }
    }
#endif

    uint64_t next_value()
    {
        return ++timeline_value_;
    }

    const OrtInteropApi& interop_;
    VkHelper::Device& device_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    NativeHandle native_handle_{};
    OrtExternalResourceImporter* importer_ = nullptr;
    OrtExternalSemaphoreHandle* ort_semaphore_ = nullptr;
    uint64_t timeline_value_ = 0;
};

static void buffer_barrier(VkCommandBuffer cmd, const VulkanBuffer& buffer, VkAccessFlags src_access,
                           VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.buffer;
    barrier.offset = 0;
    barrier.size = buffer.size;

    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

// ============================================================================
// Pipeline  (diffusion → post-processing → VAE decoder)
// ============================================================================

uint64_t run_pipeline(Ort::Session& text_encoder_session, Ort::Session& transformer_session,
                      Ort::Session& vae_decoder_session, Ort::IoBinding& text_encoder_io,
                      Ort::IoBinding& transformer_io, Ort::IoBinding& vae_decoder_io,
                      const std::vector<float>& time_schedule, VulkanBuffer& timestep_buf, VkHelper& vk,
                      OrtVulkanTimelineSemaphore& sync_semaphore, Ort::SyncStream& ort_stream,
                      VkCommandBuffer cmd_upload_lat, VkCommandBuffer* cmd_upload_time, VkCommandBuffer* cmd_euler,
                      VkCommandBuffer cmd_postprocess, ComputePipelineResources& euler_shader,
                      ComputePipelineResources& postprocess_shader, VulkanBuffer& hidden_states_buf,
                      VulkanBuffer& transformer_output_buf, VulkanBuffer& decoder_latent_buf, StagingBuffer& staging,
                      bool encode_prompt)
{
    nvtx3::scoped_range nvtx_pipeline("run_pipeline");

    Ort::RunOptions run_options;
    run_options.SetSyncStream(ort_stream);
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");

    // --- Text Encoder ---
    if (encode_prompt)
    {
        nvtx3::scoped_range nvtx("text_encoder");
        text_encoder_session.Run(run_options, text_encoder_io);
    }
    {
        nvtx3::scoped_range nvtx("upload_latent");

        vkResetCommandBuffer(cmd_upload_lat, 0);
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_upload_lat, &beginInfo));

        VkBufferCopy region = {0, 0, hidden_states_buf.size};
        vkCmdCopyBuffer(cmd_upload_lat, staging.buffer, hidden_states_buf.buffer, 1, &region);

        VK_CHECK(vkEndCommandBuffer(cmd_upload_lat));

        const uint64_t upload_done = sync_semaphore.submit_vulkan(vk, cmd_upload_lat);
        sync_semaphore.wait_ort(ort_stream, upload_done);
    }

    // --- Diffusion Loop (rectified-flow Euler) ---
    {
        vkResetCommandBuffer(cmd_upload_time[0], 0);
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_upload_time[0], &beginInfo));
        vkCmdUpdateBuffer(cmd_upload_time[0], timestep_buf.buffer, 0, sizeof(float), &time_schedule[0]);
        VK_CHECK(vkEndCommandBuffer(cmd_upload_time[0]));

        const uint64_t timestep_done = sync_semaphore.submit_vulkan(vk, cmd_upload_time[0]);
        sync_semaphore.wait_ort(ort_stream, timestep_done);
    }

    for (int step = 0; step < FLOW_STEPS; ++step)
    {
        nvtx3::scoped_range nvtx_step("diffusion_step_" + std::to_string(step));

        float t_curr = time_schedule[step];
        float t_next = time_schedule[step + 1];

        {
            nvtx3::scoped_range nvtx_transformer("transformer");
            transformer_session.Run(run_options, transformer_io);
        }

        {
            nvtx3::scoped_range nvtx_euler("euler_step");

            vkResetCommandBuffer(cmd_euler[step], 0);
            VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cmd_euler[step], &beginInfo));

            buffer_barrier(cmd_euler[step], hidden_states_buf, VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            buffer_barrier(cmd_euler[step], transformer_output_buf, VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            vkCmdBindPipeline(cmd_euler[step], VK_PIPELINE_BIND_POINT_COMPUTE, euler_shader.pipeline);
            vkCmdBindDescriptorSets(cmd_euler[step], VK_PIPELINE_BIND_POINT_COMPUTE, euler_shader.pipelineLayout, 0, 1,
                                    &euler_shader.descriptorSet, 0, nullptr);

            EulerPushConstants push_data{};
            push_data.t_curr = t_curr;
            push_data.t_next = t_next;
            push_data.total_elements = static_cast<uint32_t>(hidden_states_buf.size / sizeof(float));
            vkCmdPushConstants(cmd_euler[step], euler_shader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(EulerPushConstants), &push_data);

            uint32_t group_count_x = (push_data.total_elements + 255) / 256;
            vkCmdDispatch(cmd_euler[step], group_count_x, 1, 1);

            buffer_barrier(cmd_euler[step], hidden_states_buf, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

            if (step + 1 < FLOW_STEPS)
            {
                vkCmdUpdateBuffer(cmd_euler[step], timestep_buf.buffer, 0, sizeof(float), &t_next);
            }

            VK_CHECK(vkEndCommandBuffer(cmd_euler[step]));

            const uint64_t transformer_done = sync_semaphore.signal_ort(ort_stream);
            const uint64_t euler_done = sync_semaphore.submit_vulkan(vk, cmd_euler[step], transformer_done);
            sync_semaphore.wait_ort(ort_stream, euler_done);
        }
    }

    // --- Post-process Latents (Vulkan compute) ---
    {
        nvtx3::scoped_range nvtx_post("postprocess");

        vkResetCommandBuffer(cmd_postprocess, 0);
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_postprocess, &beginInfo));

        buffer_barrier(cmd_postprocess, hidden_states_buf, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        buffer_barrier(cmd_postprocess, decoder_latent_buf, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd_postprocess, VK_PIPELINE_BIND_POINT_COMPUTE, postprocess_shader.pipeline);
        vkCmdBindDescriptorSets(cmd_postprocess, VK_PIPELINE_BIND_POINT_COMPUTE, postprocess_shader.pipelineLayout, 0,
                                1, &postprocess_shader.descriptorSet, 0, nullptr);
        PostprocessPushConstants pp_pc{};
        pp_pc.C = static_cast<uint32_t>(LATENT_CHANNELS);
        pp_pc.I = static_cast<uint32_t>(LATENT_HEIGHT);
        pp_pc.J = static_cast<uint32_t>(LATENT_WIDTH);
        pp_pc.pi = static_cast<uint32_t>(PATCH_SIZE);
        pp_pc.pj = static_cast<uint32_t>(PATCH_SIZE);
        pp_pc.total_elements = static_cast<uint32_t>((LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE) *
                                                     (LATENT_HEIGHT * PATCH_SIZE) * (LATENT_WIDTH * PATCH_SIZE));
        vkCmdPushConstants(cmd_postprocess, postprocess_shader.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pp_pc), &pp_pc);
        vkCmdDispatch(cmd_postprocess, (pp_pc.total_elements + 255) / 256, 1, 1);

        buffer_barrier(cmd_postprocess, decoder_latent_buf, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        VK_CHECK(vkEndCommandBuffer(cmd_postprocess));

        const uint64_t transformer_done = sync_semaphore.signal_ort(ort_stream);
        const uint64_t postprocess_done = sync_semaphore.submit_vulkan(vk, cmd_postprocess, transformer_done);
        sync_semaphore.wait_ort(ort_stream, postprocess_done);
    }

    // --- VAE Decoder ---
    {
        nvtx3::scoped_range nvtx_vae("vae_decoder");

        vae_decoder_session.Run(run_options, vae_decoder_io);
    }

    return sync_semaphore.signal_ort(ort_stream);
}

struct VkPipelineState
{
    explicit VkPipelineState(Ort::Env& environment)
        : env(environment)
    {
    }

    ~VkPipelineState()
    {
        if (vk && vk->device.valid())
        {
            vk->device.deviceWaitIdle();
        }

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
        sync_semaphore.reset();
        sync_stream.reset();
        graphics_interop.reset();
        cig_external_compute_queue_data.clear();

        euler_shader.cleanup();
        postprocess_shader.cleanup();
        vk.reset();
    }

    std::unique_ptr<VkHelper> vk;
    Ort::Env& env;
    bool prompt_embeds_valid = false;
    bool use_cig = false;
    Ort::ConstEpDevice trt_device{};
    std::optional<Ort::SyncStream> sync_stream;
    const OrtInteropApi* interop_api = nullptr;
    std::vector<uint8_t> cig_external_compute_queue_data;
    std::unique_ptr<OrtVulkanGraphicsInteropScope> graphics_interop;

    std::unique_ptr<din::common::OrtRunner> text_encoder_runner;
    std::unique_ptr<din::common::OrtRunner> transformer_runner;
    std::unique_ptr<din::common::OrtRunner> vae_decoder_runner;
    std::unique_ptr<OrtVulkanTensorImporter> tensor_importer;
    std::unique_ptr<OrtVulkanTimelineSemaphore> sync_semaphore;

    VulkanBuffer token_buf{};
    VulkanBuffer attn_mask_buf{};
    VulkanBuffer text_encoder_embeds{};
    VulkanBuffer timestep_buf{};
    VulkanBuffer hidden_states{};
    VulkanBuffer img_ids_buf{};
    VulkanBuffer txt_ids_buf{};
    VulkanBuffer transformer_output{};
    VulkanBuffer decoder_latent{};
    VulkanBuffer decoded_image_buf{};
    VulkanBuffer bn_mean_buf{};
    VulkanBuffer bn_std_buf{};
    StagingBuffer staging{};
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

    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, FLOW_STEPS> cmd_bufs_euler{};
    std::array<VkCommandBuffer, FLOW_STEPS> cmd_bufs_time{};
    VkCommandBuffer cmd_upload_lat = VK_NULL_HANDLE;
    VkCommandBuffer cmd_download = VK_NULL_HANDLE;
    VkCommandBuffer cmd_postprocess = VK_NULL_HANDLE;

    ComputePipelineResources euler_shader;
    ComputePipelineResources postprocess_shader;
    std::unique_ptr<Ort::IoBinding> text_encoder_io;
    std::unique_ptr<Ort::IoBinding> transformer_io;
    std::unique_ptr<Ort::IoBinding> vae_decoder_io;
    std::vector<float> time_schedule;
};

static void initialize_vk_state(VkPipelineState& state, const Flux2Config& config, Ort::ConstEpDevice trt_device)
{
    if (config.provider != Flux2ExecutionProvider::TrtRtx)
    {
        throw std::runtime_error("Vulkan processing requires --provider trt-rtx");
    }
    const Flux2ModelPaths model_paths = MakeFlux2ModelPaths(config.model_dir, config.precision);
    state.use_cig = config.processing == Flux2ProcessingBackend::VkCig;
    const Flux2ModelCachePaths cache_paths = MakeFlux2ModelCachePaths(config.precision, state.use_cig ? "vk_cig" : "vk");

    std::cout << "Model dir: " << model_paths.base_dir.string() << "\n"
              << "CIG:       " << (state.use_cig ? "enabled" : "disabled") << "\n" << std::endl;

    // -----------------------------------------------------------------
    // Init Vulkan
    // -----------------------------------------------------------------
    std::cout << "\n=== Initializing Vulkan ===" << std::endl;
    auto nvtx_scope_vk = nvtx3::start_range("init_vulkan");

    state.vk = std::make_unique<VkHelper>(0, state.use_cig);
    std::cout << "Vulkan device initialized" << std::endl;

    nvtx3::end_range(nvtx_scope_vk);

    // -----------------------------------------------------------------
    // ORT environment & session options
    // -----------------------------------------------------------------
    state.trt_device = trt_device;
    state.interop_api = &Ort::GetInteropApi();
    if (state.use_cig)
    {
        state.cig_external_compute_queue_data = state.vk->createCudaGraphicsInteropData();
        state.graphics_interop = std::make_unique<OrtVulkanGraphicsInteropScope>(
            *state.interop_api, state.trt_device, state.cig_external_compute_queue_data);
    }
    state.sync_stream.emplace(din::common::CreateTensorRTRTXComputeStream(state.env));

    // -----------------------------------------------------------------
    // Load ONNX models
    // -----------------------------------------------------------------
    std::cout << "=== Loading ONNX Models ===" << std::endl;
    auto nvtx_scope_load = nvtx3::start_range("load_onnx_models");
    din::common::EpContextOptions ep_context;
    ep_context.output_dir = config.ep_context_dir.string();
    const std::string cache_dir = config.ep_cache_dir.string();
    std::vector<std::pair<std::string, std::string>> graphics_ep_options;
    if (state.use_cig)
    {
        const int cuda_device_ordinal = din::common::ChooseCudaDeviceOrdinal(state.trt_device);
        const auto shared_memory_info =
            din::common::QueryCudaGraphicsInteropSharedMemoryInfo(cuda_device_ordinal, false);
        if (!shared_memory_info.supports_simultaneous_graphics_compute)
        {
            throw std::runtime_error(std::string("Vulkan CIG is not supported on CUDA device generation ") +
                                     din::common::ToString(shared_memory_info.generation));
        }
        std::cout << "CUDA device ordinal " << shared_memory_info.cuda_device_ordinal
                  << " graphics interop shared memory limit: "
                  << (shared_memory_info.max_shared_memory_bytes / 1024) << " KiB" << std::endl;
        graphics_ep_options.emplace_back("nv_max_shared_mem_size",
                                         std::to_string(shared_memory_info.max_shared_memory_bytes));
        graphics_ep_options.emplace_back("nv_length_aux_stream_array", "0");
        graphics_ep_options.emplace_back("nv_use_sync_gpu_allocator", "1");
    }

    auto make_profile = [&](std::string cache_subpath)
    {
        din::common::ModelProfile profile;
        profile.cache_subpath = std::move(cache_subpath);
        profile.embed_ep_context = false;
        profile.enable_cuda_graph = !state.use_cig;
        profile.extra_ep_options = graphics_ep_options;
        return profile;
    };

    state.text_encoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.text_encoder_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.text_encoder), &*state.sync_stream);
    std::cout << "  Text encoder loaded" << std::endl;

    state.transformer_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.transformer_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.transformer), &*state.sync_stream);
    std::cout << "  Transformer loaded" << std::endl;

    state.vae_decoder_runner = std::make_unique<din::common::OrtRunner>(
        state.env, model_paths.vae_decoder_model.string(), "trt-rtx", cache_dir, ep_context,
        make_profile(cache_paths.vae_decoder), &*state.sync_stream);
    std::cout << "  VAE decoder loaded" << std::endl;

    nvtx3::end_range(nvtx_scope_load);

    // -----------------------------------------------------------------
    // Allocate tensors from Vulkan buffers imported through ORT's external
    // resource importer.
    // -----------------------------------------------------------------

    // Text encoder inputs (GPU — uploaded once via staging)
    state.tensor_importer =
        std::make_unique<OrtVulkanTensorImporter>(*state.interop_api, state.trt_device, state.vk->device);

    std::vector<int64_t> token_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    state.token_buf = state.vk->createExternalBuffer(shape_numel(token_shape) * sizeof(int64_t));
    state.token_tensor =
        state.tensor_importer->create_tensor(state.token_buf, token_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    std::vector<int64_t> attn_mask_shape = {BATCH_SIZE, SEQUENCE_LENGTH};
    state.attn_mask_buf = state.vk->createExternalBuffer(shape_numel(attn_mask_shape) * sizeof(int64_t));
    state.attention_mask_tensor =
        state.tensor_importer->create_tensor(state.attn_mask_buf, attn_mask_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    // Text encoder output (GPU)
    std::vector<int64_t> te_out_shape = {BATCH_SIZE, SEQUENCE_LENGTH, TEXT_ENCODER_EMBED_DIM};
    state.text_encoder_embeds = state.vk->createExternalBuffer(shape_numel(te_out_shape) * sizeof(float));
    state.text_encoder_embeds_tensor = state.tensor_importer->create_tensor(state.text_encoder_embeds, te_out_shape,
                                                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Timestep (GPU — updated per step via vkCmdUpdateBuffer)
    std::vector<int64_t> timestep_shape = {BATCH_SIZE};
    state.timestep_buf = state.vk->createExternalBuffer(sizeof(float));
    state.timestep_tensor =
        state.tensor_importer->create_tensor(state.timestep_buf, timestep_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Transformer hidden states (GPU)
    std::vector<int64_t> hidden_shape = {BATCH_SIZE, TRANSFORMER_HIDDEN_DIM, LATENT_CHANNELS};
    state.hidden_states = state.vk->createExternalBuffer(shape_numel(hidden_shape) * sizeof(float));
    state.hidden_count = shape_numel(hidden_shape);
    state.hidden_states_tensor =
        state.tensor_importer->create_tensor(state.hidden_states, hidden_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Position IDs (GPU — uploaded once, never change)
    std::vector<int64_t> img_ids_shape = {BATCH_SIZE, IMAGE_SEQUENCE, 4};
    state.img_ids_buf = state.vk->createExternalBuffer(shape_numel(img_ids_shape) * sizeof(int64_t));
    state.img_ids_tensor =
        state.tensor_importer->create_tensor(state.img_ids_buf, img_ids_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    std::vector<int64_t> txt_ids_shape = {BATCH_SIZE, SEQUENCE_LENGTH, 4};
    state.txt_ids_buf = state.vk->createExternalBuffer(shape_numel(txt_ids_shape) * sizeof(int64_t));
    state.txt_ids_tensor =
        state.tensor_importer->create_tensor(state.txt_ids_buf, txt_ids_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    // Transformer output (GPU — scratch for Euler step)
    state.transformer_output = state.vk->createExternalBuffer(shape_numel(hidden_shape) * sizeof(float));
    state.transformer_output_tensor = state.tensor_importer->create_tensor(state.transformer_output, hidden_shape,
                                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // VAE decoder input (GPU)
    std::vector<int64_t> dec_latent_shape = {BATCH_SIZE, LATENT_CHANNELS / PATCH_SIZE / PATCH_SIZE,
                                             LATENT_HEIGHT * PATCH_SIZE, LATENT_WIDTH * PATCH_SIZE};
    state.decoder_latent = state.vk->createExternalBuffer(shape_numel(dec_latent_shape) * sizeof(float));
    state.decoder_input_tensor = state.tensor_importer->create_tensor(state.decoder_latent, dec_latent_shape,
                                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // VAE decoder output (GPU — downloaded to CPU for image saving)
    std::vector<int64_t> image_shape = {BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH};
    state.decoded_image_bytes = shape_numel(image_shape) * sizeof(float);
    state.decoded_image_buf = state.vk->createExternalBuffer(state.decoded_image_bytes);
    state.decoded_image_tensor =
        state.tensor_importer->create_tensor(state.decoded_image_buf, image_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // -----------------------------------------------------------------
    // Staging buffer (host-visible, sized for the largest upload)
    // -----------------------------------------------------------------
    size_t staging_size = std::max(state.hidden_states.size, state.decoded_image_bytes);
    staging_size = std::max(staging_size, std::max(state.img_ids_buf.size, state.token_buf.size));
    state.staging = state.vk->createStagingBuffer(staging_size);

    // -----------------------------------------------------------------
    // Command pool (needed for one-time uploads below and for the pipeline)
    // -----------------------------------------------------------------
    state.command_pool =
        state.vk->device.createCommandPool(state.vk->queueFamilyIndex, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    // -----------------------------------------------------------------
    // Load Shaders
    // -----------------------------------------------------------------
    std::cout << "\n=== Load Shaders ===" << std::endl;

    state.euler_shader = loadEulerShader(EULER_SHADER_PATH, *state.vk, state.hidden_states, state.transformer_output);

    state.bn_mean_buf = state.vk->createExternalBuffer(LATENT_CHANNELS * sizeof(float));
    state.bn_std_buf = state.vk->createExternalBuffer(LATENT_CHANNELS * sizeof(float));

    {
        VkCommandBuffer cmd = state.vk->beginCommandBuffer(state.command_pool);
        memcpy(state.staging.mapped, BN_MEAN, LATENT_CHANNELS * sizeof(float));
        VkBufferCopy region = {0, 0, state.bn_mean_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.bn_mean_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);
    }
    {
        VkCommandBuffer cmd = state.vk->beginCommandBuffer(state.command_pool);
        memcpy(state.staging.mapped, BN_STD, LATENT_CHANNELS * sizeof(float));
        VkBufferCopy region = {0, 0, state.bn_std_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.bn_std_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);
    }

    state.postprocess_shader =
        loadComputeShader(POSTPROCESS_SHADER_PATH, *state.vk, sizeof(PostprocessPushConstants),
                          {&state.hidden_states, &state.decoder_latent, &state.bn_mean_buf, &state.bn_std_buf});

    // -----------------------------------------------------------------
    // Create timeline semaphore and import it into ORT.
    // We manage a monotonically increasing timeline counter and
    // hand it off alternately between ORT and Vulkan:
    //   ORT signals N+1 -> Vulkan waits N+1, signals N+2 -> ORT waits N+2
    // -----------------------------------------------------------------
    state.sync_semaphore =
        std::make_unique<OrtVulkanTimelineSemaphore>(*state.interop_api, state.trt_device, state.vk->device);

    // -----------------------------------------------------------------
    // Persistent command buffers (pool already created above)
    // -----------------------------------------------------------------

    // Euler sampler
    state.vk->device.allocateCommandBuffers(state.command_pool, FLOW_STEPS, state.cmd_bufs_euler.data());

    // Time scheduler
    state.vk->device.allocateCommandBuffers(state.command_pool, FLOW_STEPS, state.cmd_bufs_time.data());

    VkCommandBuffer cmd_bufs[3];
    state.vk->device.allocateCommandBuffers(state.command_pool, 3, cmd_bufs);
    state.cmd_upload_lat = cmd_bufs[0];
    state.cmd_download = cmd_bufs[1];
    state.cmd_postprocess = cmd_bufs[2];

    nvtx3::end_range(nvtx_scope_vk);

    // -----------------------------------------------------------------
    // Create persistent IoBindings (all GPU tensors, reused every run)
    // -----------------------------------------------------------------
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

    // -----------------------------------------------------------------
    // One-time upload: position IDs (via staging buffer)
    // -----------------------------------------------------------------
    {
        std::vector<int64_t> img_ids_cpu(shape_numel(img_ids_shape));
        for (int64_t b = 0; b < BATCH_SIZE; ++b)
        {
            int64_t idx = 0;
            for (int64_t h = 0; h < LATENT_HEIGHT; ++h)
                for (int64_t w = 0; w < LATENT_WIDTH; ++w, ++idx)
                {
                    int64_t base = (b * IMAGE_SEQUENCE + idx) * 4;
                    img_ids_cpu[base + 0] = 0;
                    img_ids_cpu[base + 1] = h;
                    img_ids_cpu[base + 2] = w;
                    img_ids_cpu[base + 3] = 0;
                }
        }

        memcpy(state.staging.mapped, img_ids_cpu.data(), state.img_ids_buf.size);

        VkCommandBuffer cmd = state.vk->beginCommandBuffer(state.command_pool);
        VkBufferCopy region = {0, 0, state.img_ids_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.img_ids_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);
    }
    {
        std::vector<int64_t> txt_ids_cpu(shape_numel(txt_ids_shape));
        for (int64_t b = 0; b < BATCH_SIZE; ++b)
            for (int64_t t = 0; t < SEQUENCE_LENGTH; ++t)
            {
                int64_t base = (b * SEQUENCE_LENGTH + t) * 4;
                txt_ids_cpu[base + 0] = 0;
                txt_ids_cpu[base + 1] = 0;
                txt_ids_cpu[base + 2] = 0;
                txt_ids_cpu[base + 3] = t;
            }
        memcpy(state.staging.mapped, txt_ids_cpu.data(), state.txt_ids_buf.size);

        VkCommandBuffer cmd = state.vk->beginCommandBuffer(state.command_pool);
        VkBufferCopy region = {0, 0, state.txt_ids_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.txt_ids_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);
    }

    // -----------------------------------------------------------------
    // Time schedule (computed once, constant across images)
    // -----------------------------------------------------------------
    constexpr double MU = 2.291179894115571;
    state.time_schedule.resize(FLOW_STEPS + 1);
    for (int i = 0; i <= FLOW_STEPS; ++i)
    {
        double t = T_START - (T_START - T_END) * static_cast<double>(i) / FLOW_STEPS;
        state.time_schedule[i] = static_cast<float>(std::exp(MU) / (std::exp(MU) + std::pow(1.0 / t - 1.0, 1.0)));
        std::cout << "  sigma[" << i << "] = " << state.time_schedule[i] << std::endl;
    }

    // -----------------------------------------------------------------
    // Load prompt tokens & upload to GPU (once for all images)
    // -----------------------------------------------------------------
    std::cout << "\n=== Tokenizing Prompt ===" << std::endl;
    {
        nvtx3::scoped_range nvtx("tokenize");

        const Flux2TextEncoderInputs text_inputs = TokenizeFlux2Prompt(model_paths, config.prompt);

        std::vector<int64_t> tokens_cpu(shape_numel(token_shape));
        std::vector<int64_t> attn_mask_cpu(shape_numel(attn_mask_shape));
        FillTextEncoderInputs(text_inputs.token_ids, text_inputs.pad_token_id, tokens_cpu.data(), attn_mask_cpu.data(),
                              BATCH_SIZE, SEQUENCE_LENGTH);

        memcpy(state.staging.mapped, tokens_cpu.data(), state.token_buf.size);
        VkCommandBuffer cmd = state.vk->beginCommandBuffer(state.command_pool);
        VkBufferCopy region = {0, 0, state.token_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.token_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);

        memcpy(state.staging.mapped, attn_mask_cpu.data(), state.attn_mask_buf.size);
        cmd = state.vk->beginCommandBuffer(state.command_pool);
        region = {0, 0, state.attn_mask_buf.size};
        vkCmdCopyBuffer(cmd, state.staging.buffer, state.attn_mask_buf.buffer, 1, &region);
        state.vk->endAndSubmitCommandBuffer(cmd, state.command_pool);
    }
}

static Flux2Image run_vk_image(VkPipelineState& state, unsigned int seed)
{
    initialize_latent(static_cast<float*>(state.staging.mapped), state.hidden_count, seed);

    const uint64_t vae_done = run_pipeline(
        state.text_encoder_runner->session, state.transformer_runner->session, state.vae_decoder_runner->session,
        *state.text_encoder_io, *state.transformer_io, *state.vae_decoder_io, state.time_schedule, state.timestep_buf,
        *state.vk, *state.sync_semaphore, *state.sync_stream, state.cmd_upload_lat, state.cmd_bufs_time.data(),
        state.cmd_bufs_euler.data(), state.cmd_postprocess, state.euler_shader, state.postprocess_shader,
        state.hidden_states, state.transformer_output, state.decoder_latent, state.staging, !state.prompt_embeds_valid);
    state.prompt_embeds_valid = true;

    vkResetCommandBuffer(state.cmd_download, 0);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(state.cmd_download, &beginInfo));
    VkBufferCopy region = {0, 0, state.decoded_image_bytes};
    vkCmdCopyBuffer(state.cmd_download, state.decoded_image_buf.buffer, state.staging.buffer, 1, &region);
    VK_CHECK(vkEndCommandBuffer(state.cmd_download));

    const uint64_t download_done = state.sync_semaphore->submit_vulkan(*state.vk, state.cmd_download, vae_done);
    state.sync_semaphore->wait_vulkan(state.vk->device, download_done);

    Flux2Image image;
    image.height = static_cast<int>(IMAGE_HEIGHT);
    image.width = static_cast<int>(IMAGE_WIDTH);
    image.data.resize(state.decoded_image_bytes / sizeof(float));
    std::copy_n(static_cast<const float*>(state.staging.mapped), image.data.size(), image.data.data());
    return image;
}

class VkFlux2ProcessingPipeline final : public Flux2ProcessingPipeline
{
public:
    VkFlux2ProcessingPipeline(Flux2Config config, Flux2RuntimeContext& runtime)
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
        std::cout << "Initializing Flux2 Vulkan pipeline" << std::endl;
        state_ = std::make_unique<VkPipelineState>(runtime_.env);
        initialize_vk_state(*state_, config_, runtime_.trt_device);
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

        memcpy(state_->staging.mapped, tokens_cpu.data(), state_->token_buf.size);
        VkCommandBuffer cmd = state_->vk->beginCommandBuffer(state_->command_pool);
        VkBufferCopy region = {0, 0, state_->token_buf.size};
        vkCmdCopyBuffer(cmd, state_->staging.buffer, state_->token_buf.buffer, 1, &region);
        state_->vk->endAndSubmitCommandBuffer(cmd, state_->command_pool);

        memcpy(state_->staging.mapped, attn_mask_cpu.data(), state_->attn_mask_buf.size);
        cmd = state_->vk->beginCommandBuffer(state_->command_pool);
        region = {0, 0, state_->attn_mask_buf.size};
        vkCmdCopyBuffer(cmd, state_->staging.buffer, state_->attn_mask_buf.buffer, 1, &region);
        state_->vk->endAndSubmitCommandBuffer(cmd, state_->command_pool);
    }

    Flux2Image GenerateImage(unsigned int seed) override
    {
        if (!state_)
        {
            throw std::runtime_error("Flux2 Vulkan pipeline was not initialized");
        }
        return run_vk_image(*state_, seed);
    }

private:
    Flux2Config config_;
    Flux2RuntimeContext& runtime_;
    std::unique_ptr<VkPipelineState> state_;
};

std::unique_ptr<Flux2ProcessingPipeline> CreateFlux2VkPipeline(const Flux2Config& config, Flux2RuntimeContext& runtime)
{
    return std::make_unique<VkFlux2ProcessingPipeline>(config, runtime);
}
