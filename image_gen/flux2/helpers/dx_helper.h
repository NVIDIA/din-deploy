// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef _WIN32
#error dx_helper.h is Windows-only.
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <Windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

inline void dx_check(HRESULT hr, const char* what)
{
    if (FAILED(hr))
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s failed, HRESULT=0x%08X", what, static_cast<unsigned>(hr));
        throw std::runtime_error(msg);
    }
}

struct DxBuffer
{
    ComPtr<ID3D12Resource> resource;
    size_t size = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    D3D12_GPU_VIRTUAL_ADDRESS gpu_address() const
    {
        return resource->GetGPUVirtualAddress();
    }
};

class DxContext
{
public:
    explicit DxContext(uint64_t ort_luid)
    {
        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        if (dxgi == nullptr)
        {
            throw std::runtime_error("dxgi.dll is not available");
        }
        HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
        if (d3d12 == nullptr)
        {
            throw std::runtime_error("d3d12.dll is not available");
        }

        using PFN_D3D12CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
        using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
        auto create_factory = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgi, "CreateDXGIFactory1"));
        if (create_factory == nullptr)
        {
            throw std::runtime_error("CreateDXGIFactory1 is not available");
        }
        auto create_device = reinterpret_cast<PFN_D3D12CreateDevice>(GetProcAddress(d3d12, "D3D12CreateDevice"));
        if (create_device == nullptr)
        {
            throw std::runtime_error("D3D12CreateDevice is not available");
        }

        ComPtr<IDXGIFactory1> factory;
        dx_check(create_factory(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");

        ComPtr<IDXGIAdapter1> selected_adapter;
        uint32_t matching_adapters = 0;
        for (UINT adapter_index = 0;; ++adapter_index)
        {
            ComPtr<IDXGIAdapter1> adapter;
            const HRESULT enum_result = factory->EnumAdapters1(adapter_index, &adapter);
            if (enum_result == DXGI_ERROR_NOT_FOUND)
            {
                break;
            }
            dx_check(enum_result, "EnumAdapters1");

            DXGI_ADAPTER_DESC1 desc{};
            dx_check(adapter->GetDesc1(&desc), "IDXGIAdapter1::GetDesc1");
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
            {
                continue;
            }
            const uint64_t adapter_luid =
                (static_cast<uint64_t>(static_cast<uint32_t>(desc.AdapterLuid.HighPart)) << 32) |
                static_cast<uint32_t>(desc.AdapterLuid.LowPart);
            if (adapter_luid == ort_luid)
            {
                selected_adapter = adapter;
                ++matching_adapters;
            }
        }
        if (matching_adapters != 1)
        {
            char msg[192];
            snprintf(msg, sizeof(msg), "Expected exactly one DXGI adapter matching ORT LUID 0x%016llX, found %u",
                     static_cast<unsigned long long>(ort_luid), matching_adapters);
            throw std::runtime_error(msg);
        }
        printf("ORT LUID: 0x%016llX; selected DXGI LUID: 0x%016llX\n", static_cast<unsigned long long>(ort_luid),
               static_cast<unsigned long long>(ort_luid));

        dx_check(create_device(selected_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
                 "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        dx_check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

        slots_.resize(kSubmissionRingSize);
        for (SubmissionSlot& slot : slots_)
        {
            dx_check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&slot.allocator)),
                     "CreateCommandAllocator");
            dx_check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, slot.allocator.Get(), nullptr,
                                               IID_PPV_ARGS(&slot.command_list)),
                     "CreateCommandList");
            dx_check(slot.command_list->Close(), "Close initial command list");
        }

        dx_check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fence_event == nullptr)
        {
            throw std::runtime_error("CreateEvent failed");
        }
    }

    DxContext(const DxContext&) = delete;
    DxContext& operator=(const DxContext&) = delete;

    ~DxContext()
    {
        if (fence_event != nullptr)
        {
            CloseHandle(fence_event);
        }
    }

    DxBuffer create_default_buffer(size_t size, D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON,
                                   D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return create_buffer(size, D3D12_HEAP_TYPE_DEFAULT, initial_state, flags, D3D12_HEAP_FLAG_NONE);
    }

    DxBuffer create_shared_default_buffer(size_t size,
                                          D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON,
                                          D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return create_buffer(size, D3D12_HEAP_TYPE_DEFAULT, initial_state, flags, D3D12_HEAP_FLAG_SHARED);
    }

    DxBuffer create_upload_buffer(size_t size)
    {
        return create_buffer(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_HEAP_FLAG_NONE);
    }

    void create_upload_ring(size_t size)
    {
        for (SubmissionSlot& slot : slots_)
        {
            slot.upload_buffer = create_upload_buffer(size);
            slot.upload_offset = 0;
        }
    }

    DxBuffer create_readback_buffer(size_t size)
    {
        return create_buffer(size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_HEAP_FLAG_NONE);
    }

    void* map(DxBuffer& buffer)
    {
        void* data = nullptr;
        dx_check(buffer.resource->Map(0, nullptr, &data), "Map");
        return data;
    }

    void unmap(DxBuffer& buffer)
    {
        buffer.resource->Unmap(0, nullptr);
    }

    void upload(DxBuffer& dst_buffer, const void* data, size_t size)
    {
        begin();
        record_upload(dst_buffer, data, size);
        execute_and_wait();
    }

    void readback(DxBuffer& src_buffer, DxBuffer& readback_buffer, size_t size)
    {
        if (size > src_buffer.size || size > readback_buffer.size)
        {
            throw std::runtime_error("D3D12 readback exceeds buffer size");
        }

        begin();
        transition(src_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->CopyBufferRegion(readback_buffer.resource.Get(), 0, src_buffer.resource.Get(), 0, size);
        transition(src_buffer, D3D12_RESOURCE_STATE_COMMON);
        execute_and_wait();
    }

    void begin()
    {
        if (active_slot_ != nullptr)
        {
            throw std::runtime_error("D3D12 command list recording is already active");
        }

        SubmissionSlot& slot = slots_[next_slot_];
        next_slot_ = (next_slot_ + 1) % slots_.size();

        wait_for_value(slot.retire_value);
        slot.upload_offset = 0;

        dx_check(slot.allocator->Reset(), "Reset command allocator");
        dx_check(slot.command_list->Reset(slot.allocator.Get(), nullptr), "Reset command list");

        active_slot_ = &slot;
        command_list = slot.command_list;
    }

    uint64_t execute()
    {
        if (active_slot_ == nullptr)
        {
            throw std::runtime_error("D3D12 command list execution requested without begin");
        }

        dx_check(command_list->Close(), "Close command list");
        ID3D12CommandList* lists[] = {command_list.Get()};
        queue->ExecuteCommandLists(1, lists);

        const uint64_t signal_value = ++fence_value;
        dx_check(queue->Signal(fence.Get(), signal_value), "Signal fence");
        active_slot_->retire_value = signal_value;
        active_slot_ = nullptr;
        command_list.Reset();
        return signal_value;
    }

    void execute_and_wait()
    {
        const uint64_t signal_value = execute();
        wait_for_value(signal_value);
    }

    void record_upload(DxBuffer& dst_buffer, const void* data, size_t size)
    {
        ensure_recording();
        if (active_slot_->upload_buffer.resource == nullptr)
        {
            throw std::runtime_error("D3D12 upload ring was not initialized");
        }
        if (size > dst_buffer.size)
        {
            throw std::runtime_error("D3D12 upload exceeds destination buffer size");
        }

        const size_t offset = active_slot_->upload_offset;
        const size_t next_offset = align_up(offset + size, 256);
        if (next_offset > active_slot_->upload_buffer.size)
        {
            throw std::runtime_error("D3D12 upload ring slot is too small");
        }

        void* mapped = map(active_slot_->upload_buffer);
        memcpy(static_cast<unsigned char*>(mapped) + offset, data, size);
        unmap(active_slot_->upload_buffer);
        active_slot_->upload_offset = next_offset;

        transition(dst_buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        command_list->CopyBufferRegion(dst_buffer.resource.Get(), 0, active_slot_->upload_buffer.resource.Get(), offset,
                                       size);
        transition(dst_buffer, D3D12_RESOURCE_STATE_COMMON);
    }

    void transition(DxBuffer& buffer, D3D12_RESOURCE_STATES new_state)
    {
        ensure_recording();
        if (buffer.state == new_state)
        {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = buffer.resource.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = buffer.state;
        barrier.Transition.StateAfter = new_state;
        command_list->ResourceBarrier(1, &barrier);
        buffer.state = new_state;
    }

    void uav_barrier(DxBuffer& buffer)
    {
        ensure_recording();
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = buffer.resource.Get();
        command_list->ResourceBarrier(1, &barrier);
    }

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12GraphicsCommandList> command_list;

private:
    struct SubmissionSlot
    {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> command_list;
        DxBuffer upload_buffer;
        size_t upload_offset = 0;
        uint64_t retire_value = 0;
    };

    static constexpr size_t kSubmissionRingSize = 3;

    DxBuffer create_buffer(size_t size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES initial_state,
                           D3D12_RESOURCE_FLAGS flags, D3D12_HEAP_FLAGS heap_flags)
    {
        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type = heap_type;
        heap_props.CreationNodeMask = 1;
        heap_props.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;

        DxBuffer buffer{};
        buffer.size = size;
        buffer.state = initial_state;
        dx_check(device->CreateCommittedResource(&heap_props, heap_flags, &desc, initial_state, nullptr,
                                                 IID_PPV_ARGS(&buffer.resource)),
                 "CreateCommittedResource");
        return buffer;
    }

    static size_t align_up(size_t value, size_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    void ensure_recording() const
    {
        if (active_slot_ == nullptr || command_list == nullptr)
        {
            throw std::runtime_error("D3D12 command list recording requested without begin");
        }
    }

    void wait_for_value(uint64_t value)
    {
        if (value == 0 || fence->GetCompletedValue() >= value)
        {
            return;
        }

        dx_check(fence->SetEventOnCompletion(value, fence_event), "SetEventOnCompletion");
        WaitForSingleObject(fence_event, INFINITE);
    }

    std::vector<SubmissionSlot> slots_;
    size_t next_slot_ = 0;
    SubmissionSlot* active_slot_ = nullptr;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
};

inline ComPtr<ID3DBlob> compile_compute_shader(const char* source, const char* entry)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;
    HRESULT hr =
        D3DCompile(source, strlen(source), nullptr, nullptr, nullptr, entry, "cs_5_0", flags, 0, &bytecode, &errors);
    if (FAILED(hr))
    {
        std::string msg = "D3DCompile failed";
        if (errors)
        {
            msg += ": ";
            msg.append(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
        }
        throw std::runtime_error(msg);
    }

    return bytecode;
}

inline ComPtr<ID3D12RootSignature> create_root_signature(ID3D12Device* device, const D3D12_ROOT_PARAMETER* params,
                                                         UINT param_count)
{
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = param_count;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errors);
    if (FAILED(hr))
    {
        std::string msg = "D3D12SerializeRootSignature failed";
        if (errors)
        {
            msg += ": ";
            msg.append(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
        }
        throw std::runtime_error(msg);
    }

    ComPtr<ID3D12RootSignature> root_signature;
    dx_check(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                         IID_PPV_ARGS(&root_signature)),
             "CreateRootSignature");
    return root_signature;
}

struct DxEulerConstants
{
    float t_curr;
    float t_next;
    uint32_t total_elements;
};

class DxEulerPipeline
{
public:
    explicit DxEulerPipeline(ID3D12Device* device)
    {
        const char* shader = R"(
cbuffer PushConstants : register(b0) {
    float t_curr;
    float t_next;
    uint total_elements;
};
RWStructuredBuffer<float> hidden_states : register(u0);
StructuredBuffer<float> transformer_output : register(t0);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= total_elements) return;
    float dt = t_next - t_curr;
    hidden_states[idx] += dt * transformer_output[idx];
}
)";

        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 3;
        params[0].Constants.ShaderRegister = 0;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 0;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[2].Descriptor.ShaderRegister = 0;

        root_signature = create_root_signature(device, params, 3);
        try
        {
            auto bytecode = compile_compute_shader(shader, "main");
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
            pso_desc.pRootSignature = root_signature.Get();
            pso_desc.CS.pShaderBytecode = bytecode->GetBufferPointer();
            pso_desc.CS.BytecodeLength = bytecode->GetBufferSize();
            dx_check(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline)),
                     "CreateComputePipelineState Euler");
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            return;
        }
    }

    void record(DxContext& dx, DxBuffer& hidden_states, DxBuffer& transformer_output, const DxEulerConstants& constants)
    {
        dx.transition(hidden_states, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx.transition(transformer_output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        dx.command_list->SetComputeRootSignature(root_signature.Get());
        dx.command_list->SetPipelineState(pipeline.Get());
        dx.command_list->SetComputeRoot32BitConstants(0, 3, &constants, 0);
        dx.command_list->SetComputeRootUnorderedAccessView(1, hidden_states.gpu_address());
        dx.command_list->SetComputeRootShaderResourceView(2, transformer_output.gpu_address());
        dx.command_list->Dispatch((constants.total_elements + 255) / 256, 1, 1);

        dx.uav_barrier(hidden_states);
        dx.transition(hidden_states, D3D12_RESOURCE_STATE_COMMON);
        dx.transition(transformer_output, D3D12_RESOURCE_STATE_COMMON);
    }

    uint64_t dispatch(DxContext& dx, DxBuffer& hidden_states, DxBuffer& transformer_output,
                      const DxEulerConstants& constants)
    {
        dx.begin();
        record(dx, hidden_states, transformer_output, constants);
        return dx.execute();
    }

private:
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> pipeline;
};

struct DxPostprocessConstants
{
    uint32_t C;
    uint32_t I;
    uint32_t J;
    uint32_t pi;
    uint32_t pj;
    uint32_t total_elements;
};

class DxPostprocessPipeline
{
public:
    explicit DxPostprocessPipeline(ID3D12Device* device)
    {
        const char* shader = R"(
cbuffer PushConstants : register(b0) {
    uint C;
    uint I;
    uint J;
    uint pi;
    uint pj;
    uint total_elements;
};
StructuredBuffer<float> input_data : register(t0);
RWStructuredBuffer<float> output_data : register(u0);
StructuredBuffer<float> bn_mean : register(t1);
StructuredBuffer<float> bn_std : register(t2);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= total_elements) return;

    uint C_out = C / (pi * pj);
    uint H_out = I * pi;
    uint W_out = J * pj;

    uint c_out = idx / (H_out * W_out);
    uint rem = idx % (H_out * W_out);
    uint i_out = rem / W_out;
    uint j_out = rem % W_out;

    uint i = i_out / pi;
    uint p = i_out % pi;
    uint j = j_out / pj;
    uint q = j_out % pj;
    uint in_c = c_out * pi * pj + p * pj + q;
    uint hw = i * J + j;

    output_data[idx] = input_data[hw * C + in_c] * bn_std[in_c] + bn_mean[in_c];
}
)";

        D3D12_ROOT_PARAMETER params[5]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 6;
        params[0].Constants.ShaderRegister = 0;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].Descriptor.ShaderRegister = 0;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[2].Descriptor.ShaderRegister = 0;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[3].Descriptor.ShaderRegister = 1;
        params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[4].Descriptor.ShaderRegister = 2;

        root_signature = create_root_signature(device, params, 5);
        auto bytecode = compile_compute_shader(shader, "main");

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_signature.Get();
        pso_desc.CS.pShaderBytecode = bytecode->GetBufferPointer();
        pso_desc.CS.BytecodeLength = bytecode->GetBufferSize();
        dx_check(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline)),
                 "CreateComputePipelineState Postprocess");
    }

    void record(DxContext& dx, DxBuffer& input, DxBuffer& output, DxBuffer& bn_mean, DxBuffer& bn_std,
                const DxPostprocessConstants& constants)
    {
        dx.transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx.transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx.transition(bn_mean, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx.transition(bn_std, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        dx.command_list->SetComputeRootSignature(root_signature.Get());
        dx.command_list->SetPipelineState(pipeline.Get());
        dx.command_list->SetComputeRoot32BitConstants(0, 6, &constants, 0);
        dx.command_list->SetComputeRootShaderResourceView(1, input.gpu_address());
        dx.command_list->SetComputeRootUnorderedAccessView(2, output.gpu_address());
        dx.command_list->SetComputeRootShaderResourceView(3, bn_mean.gpu_address());
        dx.command_list->SetComputeRootShaderResourceView(4, bn_std.gpu_address());
        dx.command_list->Dispatch((constants.total_elements + 255) / 256, 1, 1);

        dx.uav_barrier(output);
        dx.transition(input, D3D12_RESOURCE_STATE_COMMON);
        dx.transition(output, D3D12_RESOURCE_STATE_COMMON);
        dx.transition(bn_mean, D3D12_RESOURCE_STATE_COMMON);
        dx.transition(bn_std, D3D12_RESOURCE_STATE_COMMON);
    }

    uint64_t dispatch(DxContext& dx, DxBuffer& input, DxBuffer& output, DxBuffer& bn_mean, DxBuffer& bn_std,
                      const DxPostprocessConstants& constants)
    {
        dx.begin();
        record(dx, input, output, bn_mean, bn_std, constants);
        return dx.execute();
    }

private:
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> pipeline;
};
