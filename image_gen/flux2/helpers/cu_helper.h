// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CU_HELPER_H
#define CU_HELPER_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <utility>

// CUDA Driver API error checking macro
#define CU_CHECK(call)                                                                                     \
    do                                                                                                     \
    {                                                                                                      \
        CUresult err = (call);                                                                             \
        if (err != CUDA_SUCCESS)                                                                           \
        {                                                                                                  \
            const char* errStr;                                                                            \
            cuGetErrorString(err, &errStr);                                                                \
            char msg[256];                                                                                 \
            snprintf(msg, sizeof(msg), "CUDA Driver error: %s (%d) at %s:%d", errStr ? errStr : "unknown", \
                     static_cast<int>(err), __FILE__, __LINE__);                                           \
            fprintf(stderr, "%s\n", msg);                                                                  \
            throw std::runtime_error(msg);                                                                 \
        }                                                                                                  \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            char msg[256];                                                                                             \
            snprintf(msg, sizeof(msg), "CUDA error: %s (%d) at %s:%d", cudaGetErrorString(err), static_cast<int>(err), \
                     __FILE__, __LINE__);                                                                              \
            fprintf(stderr, "%s\n", msg);                                                                              \
            throw std::runtime_error(msg);                                                                             \
        }                                                                                                              \
    } while (0)

class CudaContext
{
public:
    CudaContext() = default;
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    CudaContext(CudaContext&& other) noexcept
        : context_(std::exchange(other.context_, nullptr))
    {
    }

    CudaContext& operator=(CudaContext&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            context_ = std::exchange(other.context_, nullptr);
        }
        return *this;
    }

    ~CudaContext()
    {
        destroy();
    }

    static CudaContext CreateVulkanCig(CUdevice device, void* shared_data)
    {
        if (shared_data == nullptr)
        {
            throw std::runtime_error("Cannot create CUDA CIG context without Vulkan shared data");
        }

        int supports_vulkan_cig = 0;
        CU_CHECK(cuDeviceGetAttribute(&supports_vulkan_cig, CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED, device));
        if (supports_vulkan_cig == 0)
        {
            throw std::runtime_error("CUDA device does not support Vulkan CIG contexts");
        }

        CUctxCigParam cig_param{};
        cig_param.sharedDataType = CIG_DATA_TYPE_NV_BLOB;
        cig_param.sharedData = shared_data;

        CUctxCreateParams create_params{};
        create_params.cigParams = &cig_param;

        CUcontext context = nullptr;
        CU_CHECK(cuCtxCreate(&context, &create_params, CU_CTX_SCHED_AUTO, device));

        size_t cig_enabled = 0;
        CU_CHECK(cuCtxGetLimit(&cig_enabled, CU_LIMIT_CIG_ENABLED));
        if (cig_enabled == 0)
        {
            CU_CHECK(cuCtxDestroy(context));
            throw std::runtime_error("Created CUDA context is not CIG-enabled");
        }

        return CudaContext(context);
    }

    CUcontext get() const
    {
        return context_;
    }

    class ScopedPush
    {
    public:
        explicit ScopedPush(const CudaContext& context)
            : context_(context.get())
        {
            if (context_ == nullptr)
            {
                throw std::runtime_error("Cannot push a null CUDA context");
            }

            CUcontext current = nullptr;
            CU_CHECK(cuCtxGetCurrent(&current));
            if (current != context_)
            {
                CU_CHECK(cuCtxPushCurrent(context_));
                pushed_ = true;
            }
        }

        ScopedPush(const ScopedPush&) = delete;
        ScopedPush& operator=(const ScopedPush&) = delete;

        ~ScopedPush()
        {
            if (!pushed_)
            {
                return;
            }

            CUcontext popped = nullptr;
            CUresult result = cuCtxPopCurrent(&popped);
            if (result != CUDA_SUCCESS || popped != context_)
            {
                const char* err = nullptr;
                cuGetErrorString(result, &err);
                fprintf(stderr, "Failed to pop CUDA context: %s\n", err ? err : "unexpected context");
            }
        }

    private:
        CUcontext context_ = nullptr;
        bool pushed_ = false;
    };

private:
    explicit CudaContext(CUcontext context)
        : context_(context)
    {
    }

    void destroy() noexcept
    {
        if (context_ == nullptr)
        {
            return;
        }

        CUresult result = cuCtxDestroy(context_);
        if (result != CUDA_SUCCESS)
        {
            const char* err = nullptr;
            cuGetErrorString(result, &err);
            fprintf(stderr, "Failed to destroy CUDA context: %s\n", err ? err : "unknown");
        }
        context_ = nullptr;
    }

    CUcontext context_ = nullptr;
};

class CudaDriverStream
{
public:
    CudaDriverStream()
    {
        CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));
    }

    CudaDriverStream(const CudaDriverStream&) = delete;
    CudaDriverStream& operator=(const CudaDriverStream&) = delete;

    ~CudaDriverStream()
    {
        if (stream_ == nullptr)
        {
            return;
        }

        CUresult result = cuStreamDestroy(stream_);
        if (result != CUDA_SUCCESS)
        {
            const char* err = nullptr;
            cuGetErrorString(result, &err);
            fprintf(stderr, "Failed to destroy CUDA stream: %s\n", err ? err : "unknown");
        }
    }

    CUstream get() const
    {
        return stream_;
    }
    cudaStream_t runtime() const
    {
        return reinterpret_cast<cudaStream_t>(stream_);
    }

private:
    CUstream stream_ = nullptr;
};

#endif  // CU_HELPER_H
