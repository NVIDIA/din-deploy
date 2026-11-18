// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

inline constexpr const char* kDinNvTensorRTRTXExecutionProvider = "nv_tensorrt_rtx";

#ifdef _WIN32
inline std::wstring ToOrtPathString(const std::filesystem::path& path)
{
    return path.wstring();
}
#else
inline std::string ToOrtPathString(const std::filesystem::path& path)
{
    return path.string();
}
#endif

namespace din::common
{

enum class CudaDeviceGeneration
{
    TuringOrOlder,
    AmpereOrAda,
    BlackwellOrNewer,
    Other,
};

struct CudaGraphicsInteropSharedMemoryInfo
{
    int cuda_device_ordinal = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    CudaDeviceGeneration generation = CudaDeviceGeneration::Other;
    bool supports_simultaneous_graphics_compute = false;
    int max_shared_memory_bytes = 0;
};

struct OrtGraphicsDeviceIdentity
{
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint64_t luid = 0;
};

[[nodiscard]] const char* ToString(CudaDeviceGeneration generation);
// Active CUDA context (device 0 if no context exists), UUID, SM and driver.
[[nodiscard]] std::string CurrentCudaDeviceCacheKey();

[[nodiscard]] int ChooseCudaDeviceOrdinal(Ort::ConstEpDevice ep_device,
                                          const char* override_env_var = "FLUX_CUDA_DEVICE_ID");

[[nodiscard]] OrtGraphicsDeviceIdentity ResolveOrtGraphicsDeviceIdentity(Ort::ConstEpDevice ep_device);

[[nodiscard]] CudaGraphicsInteropSharedMemoryInfo
QueryCudaGraphicsInteropSharedMemoryInfo(int cuda_device_ordinal, bool high_priority_compute_queue);

struct EpContextOptions
{
    std::string output_dir;
};

struct ModelProfile
{
    std::string min_shapes;
    std::string opt_shapes;
    std::string max_shapes;
    std::string cache_subpath;
    // CUDA graph capture requires static shapes between runs. Models with
    // per-run varying shapes (e.g. a growing KV cache) must disable it.
    bool enable_cuda_graph = true;
    // Embed the compiled TRT engine inside the EP-context ONNX. Set false for
    // engines larger than ~2 GB, which overflow the protobuf size limit when
    // embedded; the engine is then written as a sidecar file instead.
    bool embed_ep_context = true;
    bool skip_compile = false;
    std::vector<std::pair<std::string, std::string>> extra_ep_options;
};

class OrtRunner
{
public:
    OrtRunner(Ort::Env& env, const std::string& model_path, const std::string& provider, const std::string& cache_dir,
              const EpContextOptions& ep_context, const ModelProfile& profile = {},
              Ort::SyncStream* compute_stream_in = nullptr);

    [[nodiscard]] bool HasDeviceIo() const;
    [[nodiscard]] bool UsesUnifiedMemory() const;
    [[nodiscard]] Ort::UnownedAllocator DeviceAllocator();
    [[nodiscard]] Ort::UnownedAllocator PinnedAllocator();
    [[nodiscard]] Ort::ConstMemoryInfo DeviceMemory();

    Ort::Env& env;
    Ort::Session session{nullptr};
    Ort::SyncStream* compute_stream = nullptr;
    Ort::ConstEpDevice ep_device{};

private:
    Ort::SessionOptions options_;
    Ort::SyncStream owned_compute_stream_{nullptr};
    bool uses_unified_memory_ = false;
};

Ort::ConstEpDevice RegisterTensorRTRTXProvider(Ort::Env& env);
Ort::SyncStream CreateTensorRTRTXComputeStream(Ort::Env& env);

template <typename T>
Ort::Value MakeHostTensor(Ort::MemoryInfo& memory, std::vector<T>& values, const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<T>(memory, values.data(), values.size(), shape.data(), shape.size());
}

inline Ort::Value MakeHostTensor(Ort::MemoryInfo& memory, std::vector<uint8_t>& values,
                                 const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<bool>(memory, reinterpret_cast<bool*>(values.data()), values.size(), shape.data(),
                                          shape.size());
}

class NotificationPtr
{
public:
    NotificationPtr(Ort::SyncStream& stream)
    {
        const auto& ep_api = *Ort::GetApi().GetEpApi();
        const OrtSyncStreamImpl* stream_impl = ep_api.SyncStream_GetImpl(stream);
        if (stream_impl == nullptr)
        {
            throw std::runtime_error("failed to get ORT SyncStream implementation");
        }
        Ort::ThrowOnError(
            stream_impl->CreateNotification(const_cast<OrtSyncStreamImpl*>(stream_impl), &ort_notification_));
    }
    NotificationPtr(std::nullptr_t) {}
    NotificationPtr(const NotificationPtr&) = delete;
    NotificationPtr(NotificationPtr&& rhs) noexcept
    {
        ort_notification_ = rhs.ort_notification_;
        rhs.ort_notification_ = nullptr;
    }
    NotificationPtr& operator=(const NotificationPtr&) = delete;
    NotificationPtr& operator=(NotificationPtr&& rhs) noexcept
    {
        NotificationPtr out{nullptr};
        ort_notification_ = rhs.ort_notification_;
        rhs.ort_notification_ = nullptr;
        return *this;
    };

    void Record()
    {
        if (ort_notification_ != nullptr)
        {
            Ort::ThrowOnError(ort_notification_->Activate(ort_notification_));
        }
    }

    void Sync()
    {
        if (ort_notification_ != nullptr)
        {
            Ort::ThrowOnError(ort_notification_->WaitOnHost(ort_notification_));
        }
    }

    ~NotificationPtr()
    {
        if (ort_notification_ != nullptr)
        {
            ort_notification_->Release(ort_notification_);
        }
    }

private:
    OrtSyncNotificationImpl* ort_notification_ = nullptr;
};

// Runtime-typed storage shared by typed adapters and metadata-driven pipelines.
// EP DEFAULT allocations are used on discrete GPUs; HOST_ACCESSIBLE allocations
// are bound directly on UMA. Owners, logical views and execution pointers stay
// stable until destruction or an explicit SwapWith().
class TensorBufferBase
{
public:
    TensorBufferBase(Ort::Env& env, Ort::ConstEpDevice device, Ort::SyncStream* stream,
                     const std::vector<int64_t>& shape, ONNXTensorElementDataType type,
                     bool use_device_io, bool disable_uma = false, size_t reserved_bytes = 0);

    Ort::Value& BindingValue();
    void* HostData();
    const void* HostData() const;
    // CUDA device pointer on GPU, CPU pointer otherwise. UMA mapping is resolved
    // once at allocation; it need not equal the CPU address returned by HostData().
    void* Data();
    const void* Data() const;
    const std::vector<int64_t>& Shape() const { return shape_; }
    ONNXTensorElementDataType Type() const { return type_; }
    size_t ByteSize() const { return bytes_; }
    size_t CapacityBytes() const { return capacity_bytes_; }
    bool UsesUnifiedMemory() const { return use_unified_memory_; }
    void CopyFromHostValue(Ort::Value& value);
    void CopyAsyncToDevice();
    // Copy only the active prefix of a reusable input buffer.
    void CopyAsyncToDevice(size_t bytes);
    NotificationPtr CopyAsyncToDeviceWithNotification();
    void CopyAsyncToHost();
    void CopyAsyncToHost(size_t bytes);
    NotificationPtr CopyAsyncToHostWithNotification();
    void CopyFrom(TensorBufferBase& other);
    void SwapWith(TensorBufferBase& other);

protected:
    size_t ElementCount() const { return count_; }

private:
    void CopyPrefix(Ort::Value& source, Ort::Value& destination, size_t bytes);
    static void Copy(Ort::Env& env, std::initializer_list<Ort::Value*> src, std::initializer_list<Ort::Value*> dst,
                     Ort::SyncStream& stream);

    // Allocators belong to the environment. External streams must outlive this
    // buffer, allowing a model session to be replaced without moving its tensors.
    Ort::Env* env_ = nullptr;
    Ort::SyncStream* compute_stream_ = nullptr;
    bool use_device_io_ = false;
    bool use_unified_memory_ = false;
    std::vector<int64_t> shape_;
    ONNXTensorElementDataType type_;
    size_t count_ = 1, bytes_ = 0, capacity_bytes_ = 0;
    void* data_ = nullptr;
    Ort::AllocatorWithDefaultOptions cpu_allocator_;
    // Storage owns capacity; views expose the requested dtype and logical shape.
    Ort::Value host_storage_{nullptr};
    Ort::Value device_storage_{nullptr};
    Ort::Value host_tensor_{nullptr};
    Ort::Value device_tensor_{nullptr};
};

template <typename T>
class TensorBuffer : public TensorBufferBase
{
public:
    TensorBuffer(OrtRunner& runner, const std::vector<int64_t>& shape, bool use_device_io, bool disable_uma = false);
    TensorBuffer(Ort::Env& env, Ort::ConstEpDevice device, Ort::SyncStream* stream, const std::vector<int64_t>& shape,
                 bool use_device_io, bool disable_uma = false);
    T* HostData();
    const T* HostData() const;
    T* Data() { return static_cast<T*>(TensorBufferBase::Data()); }
    const T* Data() const { return static_cast<const T*>(TensorBufferBase::Data()); }
    void Fill(T value);
};

inline void BindOutput(Ort::IoBinding& binding, const char* name, OrtRunner& runner, bool use_device_io)
{
    if (use_device_io)
    {
        binding.BindOutput(name, runner.DeviceMemory());
    }
    else
    {
        binding.BindOutput(name, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
}

}  // namespace din::common
