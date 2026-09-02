// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef VK_HELPER_H
#define VK_HELPER_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cu_helper.h"
#include <vulkan/vulkan.h>

#ifndef DIN_ENABLE_VULKAN_VALIDATION
#define DIN_ENABLE_VULKAN_VALIDATION 0
#endif

#if defined(VK_NV_external_compute_queue)
#define DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE 1
#else
#define DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE 0
#endif

// Convert VkResult to human-readable string
inline const char* vkResultToString(VkResult result)
{
    switch (result)
    {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_NOT_READY:
        return "VK_NOT_READY";
    case VK_TIMEOUT:
        return "VK_TIMEOUT";
    case VK_EVENT_SET:
        return "VK_EVENT_SET";
    case VK_EVENT_RESET:
        return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
        return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
        return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
        return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
        return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
        return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
        return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:
        return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_SURFACE_LOST_KHR:
        return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
        return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR:
        return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
        return "VK_ERROR_OUT_OF_DATE_KHR";
    default:
        return "VK_UNKNOWN_ERROR";
    }
}

// Error checking macros with descriptive messages
#define VK_CHECK(result)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        VkResult res = (result);                                                                                       \
        if (res != VK_SUCCESS)                                                                                         \
        {                                                                                                              \
            char msg[256];                                                                                             \
            snprintf(msg, sizeof(msg), "Vulkan error: %s (%d) at %s:%d", vkResultToString(res), static_cast<int>(res), \
                     __FILE__, __LINE__);                                                                              \
            fprintf(stderr, "%s\n", msg);                                                                              \
            throw std::runtime_error(msg);                                                                             \
        }                                                                                                              \
    } while (0)

#ifdef _WIN32
inline constexpr VkExternalMemoryHandleTypeFlagBits kVkExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kVkExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
inline constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType = cudaExternalMemoryHandleTypeOpaqueWin32;
inline constexpr cudaExternalSemaphoreHandleType kCudaExternalOpaqueSemaphoreHandleType =
    cudaExternalSemaphoreHandleTypeOpaqueWin32;
inline constexpr cudaExternalSemaphoreHandleType kCudaExternalTimelineSemaphoreHandleType =
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
inline constexpr VkExternalMemoryHandleTypeFlagBits kVkExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr VkExternalSemaphoreHandleTypeFlagBits kVkExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
inline constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType = cudaExternalMemoryHandleTypeOpaqueFd;
inline constexpr cudaExternalSemaphoreHandleType kCudaExternalOpaqueSemaphoreHandleType =
    cudaExternalSemaphoreHandleTypeOpaqueFd;
inline constexpr cudaExternalSemaphoreHandleType kCudaExternalTimelineSemaphoreHandleType =
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif

struct VulkanBuffer;
struct StagingBuffer;
struct BufferAllocation;

/**
 * VkHelper - Manages Vulkan instance, device, and queue creation for CUDA interop.
 *
 * Provides easy access to Vulkan handles and utility functions for
 * external memory/semaphore operations used in Vulkan-CUDA interoperability.
 */
struct VkHelper
{
    class Device
    {
    public:
#ifdef _WIN32
        using ExternalMemoryHandle = HANDLE;
        using ExternalSemaphoreHandle = HANDLE;
        using GetMemoryHandleFn = PFN_vkGetMemoryWin32HandleKHR;
        using GetSemaphoreHandleFn = PFN_vkGetSemaphoreWin32HandleKHR;
#else
        using ExternalMemoryHandle = int;
        using ExternalSemaphoreHandle = int;
        using GetMemoryHandleFn = PFN_vkGetMemoryFdKHR;
        using GetSemaphoreHandleFn = PFN_vkGetSemaphoreFdKHR;
#endif
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        using CreateExternalComputeQueueFn = PFN_vkCreateExternalComputeQueueNV;
        using DestroyExternalComputeQueueFn = PFN_vkDestroyExternalComputeQueueNV;
        using GetExternalComputeQueueDataFn = PFN_vkGetExternalComputeQueueDataNV;
#endif

        Device() = default;
        ~Device()
        {
            shutdown();
        }

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        void initialize(VkDevice logicalDevice, GetMemoryHandleFn getMemoryHandleFn,
                        GetSemaphoreHandleFn getSemaphoreHandleFn
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
                        ,
                        CreateExternalComputeQueueFn createExternalComputeQueueFn = nullptr,
                        DestroyExternalComputeQueueFn destroyExternalComputeQueueFn = nullptr,
                        GetExternalComputeQueueDataFn getExternalComputeQueueDataFn = nullptr
#endif
        );
        void shutdown();

        VkDevice raw() const
        {
            return handle_;
        }
        operator VkDevice() const
        {
            return handle_;
        }
        bool valid() const
        {
            return handle_ != VK_NULL_HANDLE;
        }

        VkCommandPool createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);
        void allocateCommandBuffers(VkCommandPool commandPool, uint32_t commandBufferCount,
                                    VkCommandBuffer* commandBuffers,
                                    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;
        void freeCommandBuffers(VkCommandPool commandPool, uint32_t commandBufferCount,
                                const VkCommandBuffer* commandBuffers) const;

        VulkanBuffer createExternalBuffer(const VkPhysicalDeviceMemoryProperties& memoryProperties, size_t bufferSize);
        StagingBuffer createStagingBuffer(const VkPhysicalDeviceMemoryProperties& memoryProperties, size_t bufferSize,
                                          bool persistentlyMap = true);

        VkSemaphore createTimelineSemaphore(uint64_t initialValue = 0, bool exportHandle = true);
        VkFence createFence(VkFenceCreateFlags flags = 0);

        uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memoryProperties, uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) const;
        ExternalMemoryHandle getMemoryHandle(VkDeviceMemory memory) const;
        ExternalSemaphoreHandle getSemaphoreHandle(VkSemaphore semaphore) const;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        VkExternalComputeQueueNV createExternalComputeQueue(VkQueue preferredQueue);
        void getExternalComputeQueueData(VkExternalComputeQueueNV externalQueue,
                                         VkExternalComputeQueueDataParamsNV& params, void* data) const;
#endif
        void* mapMemory(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags = 0);
        void unmapMemory(VkDeviceMemory memory);
        void waitForFences(uint32_t fenceCount, const VkFence* fences, VkBool32 waitAll, uint64_t timeout) const;
        void resetFences(uint32_t fenceCount, const VkFence* fences) const;
        void deviceWaitIdle() const;

    private:
        BufferAllocation createBufferWithMemory(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                                                size_t bufferSize, VkBufferUsageFlags usageFlags,
                                                VkMemoryPropertyFlags memoryPropertyFlags,
                                                const void* bufferCreatePNext = nullptr,
                                                const void* memoryAllocatePNext = nullptr);
        void cleanupTrackedObjects();

        VkDevice handle_ = VK_NULL_HANDLE;
        GetMemoryHandleFn vkGetMemoryHandleKHR_ = nullptr;
        GetSemaphoreHandleFn vkGetSemaphoreHandleKHR_ = nullptr;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        CreateExternalComputeQueueFn vkCreateExternalComputeQueueNV_ = nullptr;
        DestroyExternalComputeQueueFn vkDestroyExternalComputeQueueNV_ = nullptr;
        GetExternalComputeQueueDataFn vkGetExternalComputeQueueDataNV_ = nullptr;
#endif

        std::vector<VkBuffer> buffers_;
        std::vector<VkDeviceMemory> memories_;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        std::vector<VkExternalComputeQueueNV> externalComputeQueues_;
#endif
        std::vector<VkCommandPool> commandPools_;
        std::vector<VkSemaphore> semaphores_;
        std::vector<VkFence> fences_;
        std::vector<VkShaderModule> shaderModules_;
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
        std::vector<VkPipelineLayout> pipelineLayouts_;
        std::vector<VkPipeline> pipelines_;
        std::vector<VkDescriptorPool> descriptorPools_;
        std::vector<VkDeviceMemory> mappedMemories_;
    };

    // Raw Vulkan handles - accessible for interop operations
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    Device device{};
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    bool cigEnabled = false;

    // Physical device properties
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
    VkPhysicalDeviceExternalComputeQueuePropertiesNV externalComputeQueueProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV};
#endif
    uint8_t deviceUUID[VK_UUID_SIZE]{};
    CUdevice cudaDevice = -1;
    int cudaDeviceIndex = -1;

#if DIN_ENABLE_VULKAN_VALIDATION
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkDebugUtilsMessageSeverityFlagsEXT debugSeverity;
#endif

    /**
     * Constructs VkHelper with optional debug message severity configuration.
     *
     * @param vendorId ORT hardware vendor ID
     * @param deviceId ORT hardware device ID
     * @param luid ORT Windows adapter LUID
     * @param debugMessageSeverity Debug message severity flags
     *        Default: VERBOSE | INFO | WARNING | ERROR when validation is enabled
     *        Examples:
     *          - All messages: VERBOSE | INFO | WARNING | ERROR
     *          - Errors only: ERROR
     *          - Disable: 0
     */
    VkHelper(uint32_t vendorId, uint32_t deviceId, uint64_t luid, bool enableCig = false,
#if DIN_ENABLE_VULKAN_VALIDATION
             VkDebugUtilsMessageSeverityFlagsEXT debugMessageSeverity =
                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
#else
             VkDebugUtilsMessageSeverityFlagsEXT debugMessageSeverity = 0
#endif
    )
    {
        cigEnabled = enableCig;
#if DIN_ENABLE_VULKAN_VALIDATION
        debugSeverity = debugMessageSeverity;
#endif
        createInstance();
        selectPhysicalDevice(vendorId, deviceId, luid);
        initCudaDevice();
        createLogicalDevice();
        loadExtensionFunctions();
    }

    ~VkHelper()
    {
        device.shutdown();
#if DIN_ENABLE_VULKAN_VALIDATION
        if (debugMessenger != VK_NULL_HANDLE)
        {
            auto vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (vkDestroyDebugUtilsMessengerEXT)
            {
                vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
            }
        }
#endif
        if (instance != VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance, nullptr);
        }
    }

    // Non-copyable
    VkHelper(const VkHelper&) = delete;
    VkHelper& operator=(const VkHelper&) = delete;

    /**
     * Find a memory type index that satisfies the requirements.
     * Used when allocating memory for buffers/images.
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        return device.findMemoryType(memoryProperties, typeFilter, properties);
    }

    Device::ExternalMemoryHandle getMemoryHandle(VkDeviceMemory memory) const
    {
        return device.getMemoryHandle(memory);
    }

    Device::ExternalSemaphoreHandle getSemaphoreHandle(VkSemaphore semaphore) const
    {
        return device.getSemaphoreHandle(semaphore);
    }

    /**
     * Submit a command buffer, optionally synchronized via a timeline semaphore.
     * When sem == VK_NULL_HANDLE: plain fire-and-forget submit (no semaphore sync).
     * When sem is provided: waits on waitVal, signals signalVal on that semaphore.
     */
    void submitVulkanAsync(VkCommandBuffer cmd, VkSemaphore sem = VK_NULL_HANDLE, uint64_t waitVal = 0,
                           uint64_t signalVal = 0)
    {
        VkTimelineSemaphoreSubmitInfo timelineInfo = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timelineInfo.waitSemaphoreValueCount = 1;
        timelineInfo.pWaitSemaphoreValues = &waitVal;
        timelineInfo.signalSemaphoreValueCount = 1;
        timelineInfo.pSignalSemaphoreValues = &signalVal;

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        if (sem != VK_NULL_HANDLE)
        {
            submitInfo.pNext = &timelineInfo;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &sem;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &sem;
        }

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    }

    /**
     * Allocate a one-time-submit command buffer from the given pool and begin recording.
     */
    VkCommandBuffer beginCommandBuffer(VkCommandPool& commandPool)
    {
        VkCommandBuffer commandBuffer;
        device.allocateCommandBuffers(commandPool, 1, &commandBuffer, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        return commandBuffer;
    }

    /**
     * End recording, submit to queue, wait for completion, then free the command buffer.
     */
    void endAndSubmitCommandBuffer(VkCommandBuffer commandBuffer, VkCommandPool& commandPool)
    {
        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));

        device.freeCommandBuffers(commandPool, 1, &commandBuffer);
    }

    /**
     * Wait for queue to finish.
     */
    void queueWaitIdle()
    {
        VK_CHECK(vkQueueWaitIdle(queue));
    }

    /**
     * Allocate a GPU-visible buffer with external memory export (for CUDA interop).
     * Defined out-of-line after VulkanBuffer is fully declared.
     */
    VulkanBuffer createExternalBuffer(size_t bufferSize);

    /**
     * Allocate a persistently-mapped host-visible staging buffer.
     * Defined out-of-line after StagingBuffer is fully declared.
     */
    StagingBuffer createStagingBuffer(size_t bufferSize);
    std::vector<uint8_t> createCudaGraphicsInteropData();

private:
#if DIN_ENABLE_VULKAN_VALIDATION
    /**
     * Debug callback for Vulkan validation layer messages.
     * Severity flags that can be combined:
     *   - VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT (diagnostic messages)
     *   - VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT    (informational messages)
     *   - VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT (potential problems)
     *   - VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   (invalid behavior)
     */
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData)
    {
        // Set a debug point here to get the debugger to stop on the call that emitted the debug call
        const char* severity = "";
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            severity = "[ERROR]";
        }
        else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            severity = "[WARNING]";
        }
        else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        {
            severity = "[INFO]";
        }
        else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        {
            severity = "[VERBOSE]";
        }

        const char* type = "";
        if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
        {
            type = "(GENERAL)";
        }
        else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
        {
            type = "(VALIDATION)";
        }
        else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
        {
            type = "(PERFORMANCE)";
        }

        fprintf(stderr, "Vulkan %s %s: %s\n", severity, type, pCallbackData->pMessage);
        return VK_FALSE;  // Don't abort on validation errors
    }
#endif

    static bool checkLayerSupport(const char* layerName)
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const auto& layer : availableLayers)
        {
            if (strcmp(layerName, layer.layerName) == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool checkDeviceExtensionSupport(const char* extensionName) const
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        for (const auto& ext : availableExtensions)
        {
            if (strcmp(extensionName, ext.extensionName) == 0)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<const char*> getRequiredDeviceExtensions() const
    {
        // Platform handle extensions needed for CUDA interop.
        std::vector<const char*> required = {
#ifdef _WIN32
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
        };

        // These may or may not be needed depending on Vulkan version
        // They were promoted to core in 1.1/1.2, but some drivers still list them
        std::vector<const char*> optional = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
            // CIG is an explicit opt-in processing mode. Normal Vulkan must
            // not alter device creation by enabling this extension.
            cigEnabled ? VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME : nullptr,
#endif
        };

        std::vector<const char*> result;

        // Check required extensions
        for (const char* ext : required)
        {
            if (checkDeviceExtensionSupport(ext))
            {
                result.push_back(ext);
                fprintf(stdout, "  [OK] %s\n", ext);
            }
            else
            {
                fprintf(stderr, "  [MISSING] %s (REQUIRED)\n", ext);
                throw std::runtime_error(std::string("Required extension not available: ") + ext);
            }
        }

        // Add optional extensions if available
        for (const char* ext : optional)
        {
            if (ext == nullptr)
            {
                continue;
            }
            if (checkDeviceExtensionSupport(ext))
            {
                result.push_back(ext);
                fprintf(stdout, "  [OK] %s (optional, in core 1.1+)\n", ext);
            }
            else
            {
                fprintf(stdout, "  [SKIP] %s (in Vulkan core)\n", ext);
            }
        }

        return result;
    }

    // Helper to safely check if a specific instance extension is available on the system
    bool checkInstanceExtensionSupport(const char* extensionName)
    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

        for (const auto& extension : availableExtensions)
        {
            if (strcmp(extensionName, extension.extensionName) == 0)
            {
                return true;
            }
        }
        return false;
    }

    void createInstance()
    {
        // Query supported Vulkan version
        uint32_t supportedVersion = VK_API_VERSION_1_0;
        vkEnumerateInstanceVersion(&supportedVersion);
        fprintf(stdout, "Vulkan loader supports: %d.%d.%d\n", VK_VERSION_MAJOR(supportedVersion),
                VK_VERSION_MINOR(supportedVersion), VK_VERSION_PATCH(supportedVersion));

        // Use the highest version supported, but at least 1.1 for external memory
        uint32_t requestedVersion = VK_API_VERSION_1_1;  // Minimum for CUDA interop
        if (supportedVersion >= VK_API_VERSION_1_2)
        {
            requestedVersion = VK_API_VERSION_1_2;  // Preferred: timeline semaphores in core
        }
        requestedVersion = VK_API_VERSION_1_4;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "VkHelper";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = requestedVersion;

        std::vector<const char*> instanceExtensions = {};
        std::vector<const char*> validationLayers;
        const void* instancePNext = nullptr;

#if DIN_ENABLE_VULKAN_VALIDATION
        // Enable debug utils extension for validation callback
        if (checkInstanceExtensionSupport(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        {
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        const char* layer_name = "VK_LAYER_KHRONOS_validation";

        const VkBool32 setting_validate_core = VK_TRUE;
        const VkBool32 setting_validate_sync = VK_TRUE;
        const VkBool32 setting_thread_safety = VK_TRUE;
        const char* setting_debug_action[] = {"VK_DBG_LAYER_ACTION_LOG_MSG"};
        const char* setting_report_flags[] = {"info", "warn", "perf", "error", "debug"};
        const VkBool32 setting_enable_message_limit = VK_TRUE;
        const int32_t setting_duplicate_message_limit = 3;

        const VkLayerSettingEXT settings[] = {
            {layer_name, "validate_core", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &setting_validate_core},
            {layer_name, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &setting_validate_sync},
            {layer_name, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &setting_thread_safety},
            {layer_name, "debug_action", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, setting_debug_action},
            {layer_name, "report_flags", VK_LAYER_SETTING_TYPE_STRING_EXT,
             static_cast<uint32_t>(std::size(setting_report_flags)), setting_report_flags},
            {layer_name, "enable_message_limit", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &setting_enable_message_limit},
            {layer_name, "duplicate_message_limit", VK_LAYER_SETTING_TYPE_INT32_EXT, 1,
             &setting_duplicate_message_limit}};

        VkLayerSettingsCreateInfoEXT layer_settings_create_info = {};
        layer_settings_create_info.sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
        layer_settings_create_info.pNext = nullptr;
        layer_settings_create_info.settingCount = static_cast<uint32_t>(std::size(settings));
        layer_settings_create_info.pSettings = settings;

        if (checkLayerSupport(layer_name))
        {
            validationLayers.push_back(layer_name);

            // Safely check if the system supports the layer settings extension
            if (checkInstanceExtensionSupport(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME))
            {
                instanceExtensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
                instancePNext = &layer_settings_create_info;
                fprintf(stdout, "Validation layers and custom settings enabled\n");
            }
            else
            {
                // Fallback: Enable layers but skip custom settings struct to prevent crashes
                instancePNext = nullptr;
                fprintf(stdout, "Validation layers enabled (default settings used; VK_EXT_layer_settings missing)\n");
            }
        }
        else
        {
            fprintf(stderr, "Warning: Validation layers requested but not supported.\n");
        }
#endif

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pNext = instancePNext;  // Safely links settings or remains nullptr
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
        createInfo.ppEnabledExtensionNames = instanceExtensions.data();
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
        fprintf(stdout, "Vulkan instance created (API %d.%d)\n", VK_VERSION_MAJOR(requestedVersion),
                VK_VERSION_MINOR(requestedVersion));

#if DIN_ENABLE_VULKAN_VALIDATION
        // Set up debug messenger callback
        if (!validationLayers.empty() && debugSeverity != 0)
        {
            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
            debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debugCreateInfo.messageSeverity = debugSeverity;
            debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debugCreateInfo.pfnUserCallback = debugCallback;
            debugCreateInfo.pUserData = nullptr;

            auto vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

            if (vkCreateDebugUtilsMessengerEXT)
            {
                VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr, &debugMessenger));
                fprintf(stdout, "Debug messenger callback registered (severity mask: 0x%x)\n", debugSeverity);
            }
            else
            {
                fprintf(stderr, "Warning: Could not load vkCreateDebugUtilsMessengerEXT\n");
            }
        }
#endif
    }

    static const char* deviceTypeToString(VkPhysicalDeviceType type)
    {
        switch (type)
        {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "Integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "CPU (Software)";
        default:
            return "Other";
        }
    }

    void selectPhysicalDevice(uint32_t vendorId, uint32_t deviceId, uint64_t luid)
    {
        uint32_t deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
        if (deviceCount == 0)
        {
            throw std::runtime_error("No GPUs with Vulkan support found");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

        fprintf(stdout, "Available Vulkan devices:\n");
        uint32_t matchCount = 0;

        for (uint32_t i = 0; i < deviceCount; ++i)
        {
            VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &idProps;
            vkGetPhysicalDeviceProperties2(devices[i], &props2);

            uint64_t deviceLuid = 0;
#ifdef _WIN32
            if (idProps.deviceLUIDValid)
            {
                memcpy(&deviceLuid, idProps.deviceLUID, sizeof(deviceLuid));
            }
#endif
            const bool matches = props2.properties.vendorID == vendorId && props2.properties.deviceID == deviceId
#ifdef _WIN32
                                 && idProps.deviceLUIDValid && deviceLuid == luid
#endif
                ;
            fprintf(stdout, "  [%u] %s (%s), vendor=0x%04X device=0x%04X",
                    i, props2.properties.deviceName, deviceTypeToString(props2.properties.deviceType),
                    props2.properties.vendorID, props2.properties.deviceID);
#ifdef _WIN32
            if (idProps.deviceLUIDValid)
            {
                fprintf(stdout, " LUID=0x%016llX", static_cast<unsigned long long>(deviceLuid));
            }
#endif
            fprintf(stdout, "%s\n", matches ? " [ORT match]" : "");
            if (matches)
            {
                physicalDevice = devices[i];
                ++matchCount;
            }
        }

        if (matchCount != 1)
        {
            throw std::runtime_error("Expected exactly one Vulkan physical device matching the ORT device identity; "
                                     "found " + std::to_string(matchCount));
        }

        // Get basic properties
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        // Get device UUID for CUDA matching
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        externalComputeQueueProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV};
#endif

        VkPhysicalDeviceIDProperties idProps{};
        idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        idProps.pNext = &externalComputeQueueProperties;
#endif

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &idProps;

        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
        memcpy(deviceUUID, idProps.deviceUUID, VK_UUID_SIZE);

        fprintf(stdout, "Selected Vulkan device: %s (Vulkan %d.%d.%d)\n", deviceProperties.deviceName,
                VK_VERSION_MAJOR(deviceProperties.apiVersion), VK_VERSION_MINOR(deviceProperties.apiVersion),
                VK_VERSION_PATCH(deviceProperties.apiVersion));
#ifdef _WIN32
        fprintf(stdout, "ORT LUID: 0x%016llX; selected Vulkan LUID: 0x%016llX\n",
                static_cast<unsigned long long>(luid), static_cast<unsigned long long>(luid));
#endif

        // Find compute queue family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        queueFamilyIndex = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            const VkQueueFlags requiredFlags =
                VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | (cigEnabled ? VK_QUEUE_GRAPHICS_BIT : 0);
            if ((queueFamilies[i].queueFlags & requiredFlags) == requiredFlags)
            {
                queueFamilyIndex = i;
                break;
            }
        }
        if (queueFamilyIndex == UINT32_MAX)
        {
            throw std::runtime_error(cigEnabled
                                         ? "No Vulkan queue family satisfies graphics/compute/transfer CIG requirements"
                                         : "No Vulkan queue family satisfies compute/transfer requirements");
        }
    }

    void initCudaDevice()
    {
        CU_CHECK(cuInit(0));

        int cudaDeviceCount = 0;
        CU_CHECK(cuDeviceGetCount(&cudaDeviceCount));

        for (int i = 0; i < cudaDeviceCount; ++i)
        {
            CUdevice candidate = -1;
            CU_CHECK(cuDeviceGet(&candidate, i));

            CUuuid uuid{};
            CU_CHECK(cuDeviceGetUuid(&uuid, candidate));

            // Compare UUID to find matching CUDA device
            if (memcmp(uuid.bytes, deviceUUID, VK_UUID_SIZE) == 0)
            {
                char name[256]{};
                CU_CHECK(cuDeviceGetName(name, sizeof(name), candidate));

                int supportsVulkanCig = 0;
                CU_CHECK(cuDeviceGetAttribute(&supportsVulkanCig, CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED, candidate));
                if (cigEnabled && supportsVulkanCig == 0)
                {
                    throw std::runtime_error("Vulkan CIG was requested but the matched CUDA device does not support it");
                }

                cudaDevice = candidate;
                cudaDeviceIndex = i;
                fprintf(stdout, "Matched CUDA device %d: %s (Vulkan CIG %s)\n", i, name,
                        supportsVulkanCig != 0 ? "supported" : "not supported");
                return;
            }
        }

        throw std::runtime_error("No CUDA device matched the selected Vulkan device UUID");
    }

    void createLogicalDevice()
    {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Get extensions (checks availability and prints status)
        fprintf(stdout, "Checking device extensions:\n");
        std::vector<const char*> deviceExtensions = getRequiredDeviceExtensions();
        bool enableExternalComputeQueue = false;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        enableExternalComputeQueue = cigEnabled;
        if (enableExternalComputeQueue &&
            (!checkDeviceExtensionSupport(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME) ||
             externalComputeQueueProperties.maxExternalQueues == 0 || externalComputeQueueProperties.externalDataSize == 0))
        {
            throw std::runtime_error("Vulkan CIG was requested but VK_NV_external_compute_queue is unavailable");
        }
#endif

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.bufferDeviceAddress = enableExternalComputeQueue ? VK_TRUE : VK_FALSE;
        vulkan12Features.timelineSemaphore = VK_TRUE;

#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        VkExternalComputeQueueDeviceCreateInfoNV externalComputeQueueInfo{};
        externalComputeQueueInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV;
        if (enableExternalComputeQueue)
        {
            externalComputeQueueInfo.reservedExternalQueues = 1;
            externalComputeQueueInfo.pNext = &vulkan12Features;
        }
#endif

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        deviceCreateInfo.pNext = enableExternalComputeQueue ? static_cast<void*>(&externalComputeQueueInfo)
                                                            : static_cast<void*>(&vulkan12Features);
#else
        deviceCreateInfo.pNext = &vulkan12Features;
#endif

        VkDevice logicalDevice = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice));
        vkGetDeviceQueue(logicalDevice, queueFamilyIndex, 0, &queue);
        device.initialize(logicalDevice, nullptr, nullptr);
        fprintf(stdout, "Vulkan logical device created\n");
    }

    void loadExtensionFunctions()
    {
        Device::GetMemoryHandleFn getMemoryHandleFn = nullptr;
        Device::GetSemaphoreHandleFn getSemaphoreHandleFn = nullptr;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
        auto createExternalComputeQueueFn = reinterpret_cast<Device::CreateExternalComputeQueueFn>(
            vkGetDeviceProcAddr(device.raw(), "vkCreateExternalComputeQueueNV"));
        auto destroyExternalComputeQueueFn = reinterpret_cast<Device::DestroyExternalComputeQueueFn>(
            vkGetDeviceProcAddr(device.raw(), "vkDestroyExternalComputeQueueNV"));
        auto getExternalComputeQueueDataFn = reinterpret_cast<Device::GetExternalComputeQueueDataFn>(
            vkGetDeviceProcAddr(device.raw(), "vkGetExternalComputeQueueDataNV"));
#endif

#ifdef _WIN32
        getMemoryHandleFn =
            reinterpret_cast<Device::GetMemoryHandleFn>(vkGetDeviceProcAddr(device.raw(), "vkGetMemoryWin32HandleKHR"));
        getSemaphoreHandleFn = reinterpret_cast<Device::GetSemaphoreHandleFn>(
            vkGetDeviceProcAddr(device.raw(), "vkGetSemaphoreWin32HandleKHR"));
#else
        getMemoryHandleFn =
            reinterpret_cast<Device::GetMemoryHandleFn>(vkGetDeviceProcAddr(device.raw(), "vkGetMemoryFdKHR"));
        getSemaphoreHandleFn =
            reinterpret_cast<Device::GetSemaphoreHandleFn>(vkGetDeviceProcAddr(device.raw(), "vkGetSemaphoreFdKHR"));
#endif

        if (!getMemoryHandleFn || !getSemaphoreHandleFn)
        {
            throw std::runtime_error("Failed to load required Vulkan external handle functions");
        }

        device.initialize(device.raw(), getMemoryHandleFn, getSemaphoreHandleFn
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
                          ,
                          createExternalComputeQueueFn, destroyExternalComputeQueueFn, getExternalComputeQueueDataFn
#endif
        );
    }
};

// ============================================================================
// File I/O helper for loading shader bytecode
// ============================================================================

inline std::vector<uint8_t> loadBytes(const std::string& file_path)
{
    std::ifstream file_stream(file_path, std::ios::in | std::ios::binary);
    if (file_stream.fail())
    {
        std::cerr << "ERROR: Cannot open file: " << file_path << std::endl;
        std::exit(1);
    }
    size_t file_size = std::filesystem::file_size(file_path);
    std::vector<uint8_t> data(file_size);
    file_stream.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

// ============================================================================
// VULKAN BUFFER HELPER FUNCTION
// ============================================================================

struct VulkanBuffer
{
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    size_t allocationSize;
};

struct BufferAllocation
{
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t allocationSize;
};

struct ImportedExternalVKMemory
{
    cudaExternalMemory_t externalMemory;
    void* cudaPtr;
    size_t size;
};

inline ImportedExternalVKMemory importExternalVKMemory(const VkHelper::Device& device, const VulkanBuffer& buffer)
{
    auto bufferSize = buffer.size;
    auto allocationSize = buffer.allocationSize;

    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memHandleDesc.type = kCudaExternalMemoryHandleType;
#ifdef _WIN32
    HANDLE memoryHandle = device.getMemoryHandle(buffer.memory);
    if (memoryHandle == nullptr)
    {
        throw std::runtime_error("Invalid Vulkan memory Win32 handle");
    }
    memHandleDesc.handle.win32.handle = memoryHandle;
    memHandleDesc.handle.win32.name = nullptr;
#else
    int fd_memory = device.getMemoryHandle(buffer.memory);
    if (fd_memory < 0)
    {
        fprintf(stderr, "Error: Invalid file descriptor %d from Vulkan\n", fd_memory);
        throw std::runtime_error("Invalid Vulkan memory file descriptor");
    }
    memHandleDesc.handle.fd = fd_memory;
#endif
    memHandleDesc.size = allocationSize;
    memHandleDesc.flags = 0;

    cudaExternalMemory_t externalMem;
    cudaError_t importResult = cudaImportExternalMemory(&externalMem, &memHandleDesc);
#ifdef _WIN32
    CloseHandle(memoryHandle);
#endif
    CUDA_CHECK(importResult);

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = bufferSize;
    bufferDesc.flags = 0;

    void* cudaPtr = nullptr;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaPtr, externalMem, &bufferDesc));
    std::cout << "Imported Vulkan buffer into CUDA (size=" << bufferSize << " bytes, CUDA ptr=" << cudaPtr << ")"
              << std::endl;
    return {externalMem, cudaPtr, bufferSize};
}

inline cudaExternalSemaphore_t importExternalVKSemaphore(const VkHelper::Device& device, VkSemaphore vkSemaphore,
                                                         bool timelineSemaphore = true)
{
    cudaExternalSemaphoreHandleDesc semHandleDesc{};
    semHandleDesc.type =
        timelineSemaphore ? kCudaExternalTimelineSemaphoreHandleType : kCudaExternalOpaqueSemaphoreHandleType;
#ifdef _WIN32
    HANDLE semaphoreHandle = device.getSemaphoreHandle(vkSemaphore);
    if (semaphoreHandle == nullptr)
    {
        throw std::runtime_error("Invalid Vulkan semaphore Win32 handle");
    }
    semHandleDesc.handle.win32.handle = semaphoreHandle;
    semHandleDesc.handle.win32.name = nullptr;
#else
    const int fd = device.getSemaphoreHandle(vkSemaphore);
    if (fd < 0)
    {
        fprintf(stderr, "Error: Invalid semaphore file descriptor %d from Vulkan\n", fd);
        throw std::runtime_error("Invalid Vulkan semaphore file descriptor");
    }
    semHandleDesc.handle.fd = fd;
#endif
    semHandleDesc.flags = 0;

    cudaExternalSemaphore_t cudaSemaphore = nullptr;
    cudaError_t importResult = cudaImportExternalSemaphore(&cudaSemaphore, &semHandleDesc);
#ifdef _WIN32
    CloseHandle(semaphoreHandle);
#endif
    CUDA_CHECK(importResult);
    std::cout << "Imported Vulkan semaphore into CUDA" << std::endl;
    return cudaSemaphore;
}

// ============================================================================
// HOST-VISIBLE STAGING BUFFER  (persistently mapped for CPU→GPU uploads)
// ============================================================================

struct StagingBuffer
{
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped;
    size_t size;
};

inline VulkanBuffer VkHelper::createExternalBuffer(size_t bufferSize)
{
    return device.createExternalBuffer(memoryProperties, bufferSize);
}

inline StagingBuffer VkHelper::createStagingBuffer(size_t bufferSize)
{
    return device.createStagingBuffer(memoryProperties, bufferSize, true);
}

inline std::vector<uint8_t> VkHelper::createCudaGraphicsInteropData()
{
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
    if (externalComputeQueueProperties.externalDataSize == 0)
    {
        throw std::runtime_error("Vulkan external compute queue data size is zero");
    }

    VkExternalComputeQueueNV externalQueue = device.createExternalComputeQueue(queue);

    VkExternalComputeQueueDataParamsNV params{};
    params.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV;
    params.deviceIndex = 0;

    std::vector<uint8_t> data(externalComputeQueueProperties.externalDataSize);
    device.getExternalComputeQueueData(externalQueue, params, data.data());
    return data;
#else
    throw std::runtime_error("VK_NV_external_compute_queue is not available in the Vulkan headers");
#endif
}

inline void VkHelper::Device::initialize(VkDevice logicalDevice, VkHelper::Device::GetMemoryHandleFn getMemoryHandleFn,
                                         VkHelper::Device::GetSemaphoreHandleFn getSemaphoreHandleFn
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
                                         ,
                                         VkHelper::Device::CreateExternalComputeQueueFn createExternalComputeQueueFn,
                                         VkHelper::Device::DestroyExternalComputeQueueFn destroyExternalComputeQueueFn,
                                         VkHelper::Device::GetExternalComputeQueueDataFn getExternalComputeQueueDataFn
#endif
)
{
    handle_ = logicalDevice;
    if (getMemoryHandleFn != nullptr)
    {
        vkGetMemoryHandleKHR_ = getMemoryHandleFn;
    }
    if (getSemaphoreHandleFn != nullptr)
    {
        vkGetSemaphoreHandleKHR_ = getSemaphoreHandleFn;
    }
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
    if (createExternalComputeQueueFn != nullptr)
    {
        vkCreateExternalComputeQueueNV_ = createExternalComputeQueueFn;
    }
    if (destroyExternalComputeQueueFn != nullptr)
    {
        vkDestroyExternalComputeQueueNV_ = destroyExternalComputeQueueFn;
    }
    if (getExternalComputeQueueDataFn != nullptr)
    {
        vkGetExternalComputeQueueDataNV_ = getExternalComputeQueueDataFn;
    }
#endif
}

inline void VkHelper::Device::shutdown()
{
    if (handle_ == VK_NULL_HANDLE)
    {
        return;
    }
    vkDeviceWaitIdle(handle_);
    cleanupTrackedObjects();
    vkDestroyDevice(handle_, nullptr);
    handle_ = VK_NULL_HANDLE;
    vkGetMemoryHandleKHR_ = nullptr;
    vkGetSemaphoreHandleKHR_ = nullptr;
#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
    vkCreateExternalComputeQueueNV_ = nullptr;
    vkDestroyExternalComputeQueueNV_ = nullptr;
    vkGetExternalComputeQueueDataNV_ = nullptr;
#endif
}

inline void VkHelper::Device::cleanupTrackedObjects()
{
    for (auto it = mappedMemories_.rbegin(); it != mappedMemories_.rend(); ++it)
    {
        vkUnmapMemory(handle_, *it);
    }
    mappedMemories_.clear();

#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
    if (vkDestroyExternalComputeQueueNV_)
    {
        for (auto it = externalComputeQueues_.rbegin(); it != externalComputeQueues_.rend(); ++it)
        {
            vkDestroyExternalComputeQueueNV_(handle_, *it, nullptr);
        }
    }
    externalComputeQueues_.clear();
#endif

    for (auto it = fences_.rbegin(); it != fences_.rend(); ++it)
        vkDestroyFence(handle_, *it, nullptr);
    for (auto it = semaphores_.rbegin(); it != semaphores_.rend(); ++it)
        vkDestroySemaphore(handle_, *it, nullptr);
    for (auto it = pipelines_.rbegin(); it != pipelines_.rend(); ++it)
        vkDestroyPipeline(handle_, *it, nullptr);
    for (auto it = pipelineLayouts_.rbegin(); it != pipelineLayouts_.rend(); ++it)
        vkDestroyPipelineLayout(handle_, *it, nullptr);
    for (auto it = descriptorPools_.rbegin(); it != descriptorPools_.rend(); ++it)
        vkDestroyDescriptorPool(handle_, *it, nullptr);
    for (auto it = descriptorSetLayouts_.rbegin(); it != descriptorSetLayouts_.rend(); ++it)
        vkDestroyDescriptorSetLayout(handle_, *it, nullptr);
    for (auto it = shaderModules_.rbegin(); it != shaderModules_.rend(); ++it)
        vkDestroyShaderModule(handle_, *it, nullptr);
    for (auto it = commandPools_.rbegin(); it != commandPools_.rend(); ++it)
        vkDestroyCommandPool(handle_, *it, nullptr);
    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it)
        vkDestroyBuffer(handle_, *it, nullptr);
    for (auto it = memories_.rbegin(); it != memories_.rend(); ++it)
        vkFreeMemory(handle_, *it, nullptr);

    fences_.clear();
    semaphores_.clear();
    pipelines_.clear();
    pipelineLayouts_.clear();
    descriptorPools_.clear();
    descriptorSetLayouts_.clear();
    shaderModules_.clear();
    commandPools_.clear();
    buffers_.clear();
    memories_.clear();
}

inline uint32_t VkHelper::Device::findMemoryType(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                                                 uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

inline VkCommandPool VkHelper::Device::createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = flags;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(handle_, &poolInfo, nullptr, &commandPool));
    commandPools_.push_back(commandPool);
    return commandPool;
}

inline void VkHelper::Device::allocateCommandBuffers(VkCommandPool commandPool, uint32_t commandBufferCount,
                                                     VkCommandBuffer* commandBuffers, VkCommandBufferLevel level) const
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = level;
    allocInfo.commandBufferCount = commandBufferCount;
    VK_CHECK(vkAllocateCommandBuffers(handle_, &allocInfo, commandBuffers));
}

inline void VkHelper::Device::freeCommandBuffers(VkCommandPool commandPool, uint32_t commandBufferCount,
                                                 const VkCommandBuffer* commandBuffers) const
{
    vkFreeCommandBuffers(handle_, commandPool, commandBufferCount, commandBuffers);
}

inline BufferAllocation
VkHelper::Device::createBufferWithMemory(const VkPhysicalDeviceMemoryProperties& memoryProperties, size_t bufferSize,
                                         VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags,
                                         const void* bufferCreatePNext, const void* memoryAllocatePNext)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = bufferCreatePNext;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(handle_, &bufferInfo, nullptr, &buffer));
    buffers_.push_back(buffer);

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(handle_, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = memoryAllocatePNext;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memoryProperties, memRequirements.memoryTypeBits, memoryPropertyFlags);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(handle_, &allocInfo, nullptr, &memory));
    memories_.push_back(memory);
    VK_CHECK(vkBindBufferMemory(handle_, buffer, memory, 0));

    return {buffer, memory, static_cast<size_t>(memRequirements.size)};
}

inline VulkanBuffer VkHelper::Device::createExternalBuffer(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                                                           size_t bufferSize)
{
    VkExternalMemoryBufferCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalInfo.handleTypes = kVkExternalMemoryHandleType;

    VkExportMemoryAllocateInfo exportAllocInfo{};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = kVkExternalMemoryHandleType;

    const auto allocation = createBufferWithMemory(
        memoryProperties, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &externalInfo, &exportAllocInfo);

    std::cout << "Created Vulkan external buffer (size=" << bufferSize << " bytes)" << std::endl;
    return {allocation.buffer, allocation.memory, bufferSize, allocation.allocationSize};
}

inline StagingBuffer VkHelper::Device::createStagingBuffer(const VkPhysicalDeviceMemoryProperties& memoryProperties,
                                                           size_t bufferSize, bool persistentlyMap)
{
    const auto allocation = createBufferWithMemory(
        memoryProperties, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped = nullptr;
    if (persistentlyMap)
    {
        mapped = mapMemory(allocation.memory, 0, bufferSize, 0);
    }

    std::cout << "Created staging buffer (size=" << bufferSize << " bytes)" << std::endl;
    return {allocation.buffer, allocation.memory, mapped, bufferSize};
}

inline VkSemaphore VkHelper::Device::createTimelineSemaphore(uint64_t initialValue, bool exportHandle)
{
    VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = initialValue;

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = exportHandle ? kVkExternalSemaphoreHandleType : 0;
    exportInfo.pNext = &typeInfo;

    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semInfo.pNext = &exportInfo;

    VkSemaphore semaphore = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSemaphore(handle_, &semInfo, nullptr, &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
}

inline VkFence VkHelper::Device::createFence(VkFenceCreateFlags flags)
{
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = flags;

    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(handle_, &fenceInfo, nullptr, &fence));
    fences_.push_back(fence);
    return fence;
}

inline VkHelper::Device::ExternalMemoryHandle VkHelper::Device::getMemoryHandle(VkDeviceMemory memory) const
{
    if (!vkGetMemoryHandleKHR_)
    {
        throw std::runtime_error("Vulkan external memory handle function was not loaded");
    }
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR getHandleInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    getHandleInfo.memory = memory;
    getHandleInfo.handleType = kVkExternalMemoryHandleType;

    HANDLE handle = nullptr;
    VK_CHECK(vkGetMemoryHandleKHR_(handle_, &getHandleInfo, &handle));
    return handle;
#else
    VkMemoryGetFdInfoKHR getFdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    getFdInfo.memory = memory;
    getFdInfo.handleType = kVkExternalMemoryHandleType;

    int fd = -1;
    VK_CHECK(vkGetMemoryHandleKHR_(handle_, &getFdInfo, &fd));
    return fd;
#endif
}

inline VkHelper::Device::ExternalSemaphoreHandle VkHelper::Device::getSemaphoreHandle(VkSemaphore semaphore) const
{
    if (!vkGetSemaphoreHandleKHR_)
    {
        throw std::runtime_error("Vulkan external semaphore handle function was not loaded");
    }
#ifdef _WIN32
    VkSemaphoreGetWin32HandleInfoKHR getHandleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    getHandleInfo.semaphore = semaphore;
    getHandleInfo.handleType = kVkExternalSemaphoreHandleType;

    HANDLE handle = nullptr;
    VK_CHECK(vkGetSemaphoreHandleKHR_(handle_, &getHandleInfo, &handle));
    return handle;
#else
    VkSemaphoreGetFdInfoKHR getFdInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    getFdInfo.semaphore = semaphore;
    getFdInfo.handleType = kVkExternalSemaphoreHandleType;

    int fd = -1;
    VK_CHECK(vkGetSemaphoreHandleKHR_(handle_, &getFdInfo, &fd));
    return fd;
#endif
}

#if DIN_HAS_VK_NV_EXTERNAL_COMPUTE_QUEUE
inline VkExternalComputeQueueNV VkHelper::Device::createExternalComputeQueue(VkQueue preferredQueue)
{
    if (!vkCreateExternalComputeQueueNV_)
    {
        throw std::runtime_error("vkCreateExternalComputeQueueNV was not loaded");
    }

    VkExternalComputeQueueCreateInfoNV createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV;
    createInfo.preferredQueue = preferredQueue;

    VkExternalComputeQueueNV externalQueue = VK_NULL_HANDLE;
    VK_CHECK(vkCreateExternalComputeQueueNV_(handle_, &createInfo, nullptr, &externalQueue));
    externalComputeQueues_.push_back(externalQueue);
    return externalQueue;
}

inline void VkHelper::Device::getExternalComputeQueueData(VkExternalComputeQueueNV externalQueue,
                                                          VkExternalComputeQueueDataParamsNV& params, void* data) const
{
    if (!vkGetExternalComputeQueueDataNV_)
    {
        throw std::runtime_error("vkGetExternalComputeQueueDataNV was not loaded");
    }

    vkGetExternalComputeQueueDataNV_(externalQueue, &params, data);
}
#endif

inline void* VkHelper::Device::mapMemory(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                                         VkMemoryMapFlags flags)
{
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(handle_, memory, offset, size, flags, &mapped));
    if (std::find(mappedMemories_.begin(), mappedMemories_.end(), memory) == mappedMemories_.end())
    {
        mappedMemories_.push_back(memory);
    }
    return mapped;
}

inline void VkHelper::Device::unmapMemory(VkDeviceMemory memory)
{
    auto it = std::find(mappedMemories_.begin(), mappedMemories_.end(), memory);
    if (it != mappedMemories_.end())
    {
        vkUnmapMemory(handle_, memory);
        mappedMemories_.erase(it);
    }
}

inline void VkHelper::Device::waitForFences(uint32_t fenceCount, const VkFence* fences, VkBool32 waitAll,
                                            uint64_t timeout) const
{
    VK_CHECK(vkWaitForFences(handle_, fenceCount, fences, waitAll, timeout));
}

inline void VkHelper::Device::resetFences(uint32_t fenceCount, const VkFence* fences) const
{
    VK_CHECK(vkResetFences(handle_, fenceCount, fences));
}

inline void VkHelper::Device::deviceWaitIdle() const
{
    VK_CHECK(vkDeviceWaitIdle(handle_));
}

// ============================================================================
// CUDA/VULKAN ASYNC TIMELINE SEMAPHORE HELPERS
//
// These helpers wrap the boilerplate struct setup for CUDA-Vulkan semaphore
// signalling.  Callers are responsible for creating the semaphore objects and
// tracking the monotonically increasing timeline counter themselves:
//
//   VkSemaphore             vkSemaphore  = vk.device.createTimelineSemaphore(0, true);
//   cudaExternalSemaphore_t cudaSemaphore = importExternalVKSemaphore(vk.device, vkSemaphore);
//   uint64_t                semaphore_value = 0;
//
// Typical interop handoff (CUDA hands work to Vulkan, Vulkan hands back):
//
//   cudaSignalSemaphore(stream, cudaSemaphore, ++semaphore_value);          // CUDA → Vulkan
//   vk.submitVulkanAsync(cmd, vkSemaphore, semaphore_value, semaphore_value + 1);
//   ++semaphore_value;
//   cudaWaitSemaphore(stream, cudaSemaphore, semaphore_value);              // Vulkan → CUDA
// ============================================================================

// Signal a CUDA external semaphore to the given timeline value on the given stream.
inline void cudaSignalSemaphore(cudaStream_t stream, cudaExternalSemaphore_t sem, uint64_t value)
{
    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = value;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&sem, &params, 1, stream));
}

// Wait on a CUDA external semaphore for the given timeline value on the given stream.
inline void cudaWaitSemaphore(cudaStream_t stream, cudaExternalSemaphore_t sem, uint64_t value)
{
    cudaExternalSemaphoreWaitParams params = {};
    params.params.fence.value = value;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&sem, &params, 1, stream));
}

struct ComputePipelineResources
{
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    ~ComputePipelineResources()
    {
        cleanup();
    }

    ComputePipelineResources() = default;
    ComputePipelineResources(const ComputePipelineResources&) = delete;
    ComputePipelineResources& operator=(const ComputePipelineResources&) = delete;

    ComputePipelineResources(ComputePipelineResources&& other) noexcept
    {
        *this = std::move(other);
    }

    ComputePipelineResources& operator=(ComputePipelineResources&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            device = other.device;
            pipeline = other.pipeline;
            pipelineLayout = other.pipelineLayout;
            descriptorSet = other.descriptorSet;
            descriptorSetLayout = other.descriptorSetLayout;
            descriptorPool = other.descriptorPool;

            other.device = VK_NULL_HANDLE;
            other.pipeline = VK_NULL_HANDLE;
            other.pipelineLayout = VK_NULL_HANDLE;
            other.descriptorSet = VK_NULL_HANDLE;
            other.descriptorSetLayout = VK_NULL_HANDLE;
            other.descriptorPool = VK_NULL_HANDLE;
        }
        return *this;
    }

    void cleanup()
    {
        if (device == VK_NULL_HANDLE)
        {
            return;
        }
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        descriptorSet = VK_NULL_HANDLE;
    }
};

inline void destroyComputePipeline(VkDevice device, ComputePipelineResources& res)
{
    if (res.device == VK_NULL_HANDLE)
    {
        res.device = device;
    }
    res.cleanup();
}

struct EulerPushConstants
{
    float t_curr;
    float t_next;
    uint32_t total_elements;
};

struct PostprocessPushConstants
{
    uint32_t C;
    uint32_t I;
    uint32_t J;
    uint32_t pi;
    uint32_t pj;
    uint32_t total_elements;
};

inline ComputePipelineResources loadComputeShader(const std::string& shader_path, VkHelper& vk,
                                                  uint32_t push_constant_size,
                                                  const std::vector<VulkanBuffer*>& buffers)
{
    ComputePipelineResources resources{};
    resources.device = vk.device.raw();
    uint32_t num_buffers = static_cast<uint32_t>(buffers.size());

    auto spirv = loadBytes(shader_path);
    VkShaderModuleCreateInfo shader_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spirv.size();
    shader_info.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

    VkShaderModule shader_module;
    VK_CHECK(vkCreateShaderModule(vk.device, &shader_info, nullptr, &shader_module));

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = num_buffers;
    layout_info.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &resources.descriptorSetLayout));

    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = push_constant_size;

    VkPipelineLayoutCreateInfo pl_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &resources.descriptorSetLayout;
    pl_info.pushConstantRangeCount = (push_constant_size > 0) ? 1u : 0u;
    pl_info.pPushConstantRanges = (push_constant_size > 0) ? &push_range : nullptr;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &pl_info, nullptr, &resources.pipelineLayout));

    VkPipelineShaderStageCreateInfo stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader_module;
    stage.pName = "main";

    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = resources.pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &resources.pipeline));

    vkDestroyShaderModule(vk.device, shader_module, nullptr);

    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_buffers};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    VK_CHECK(vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &resources.descriptorPool));

    VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = resources.descriptorPool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &resources.descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &alloc_info, &resources.descriptorSet));

    std::vector<VkDescriptorBufferInfo> buf_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i)
    {
        buf_infos[i] = {buffers[i]->buffer, 0, buffers[i]->size};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = resources.descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(vk.device, num_buffers, writes.data(), 0, nullptr);

    std::cout << "  Loaded " << shader_path << " (" << num_buffers << " bindings)" << std::endl;
    return resources;
}

inline ComputePipelineResources loadEulerShader(const std::string& shader_path, VkHelper& vk,
                                                VulkanBuffer& hidden_states, const VulkanBuffer& transformer_output)
{
    ComputePipelineResources resources{};
    resources.device = vk.device.raw();

    // ============================================================================
    // LOAD VULKAN SHADER
    // ============================================================================
    std::cout << "\n=== Loading Euler Shader ===" << std::endl;

    auto euler_spirv = loadBytes(shader_path);
    VkShaderModuleCreateInfo euler_shader_info = {};
    euler_shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    euler_shader_info.codeSize = euler_spirv.size();
    euler_shader_info.pCode = reinterpret_cast<const uint32_t*>(euler_spirv.data());

    VkShaderModule euler_shader_module;
    VK_CHECK(vkCreateShaderModule(vk.device, &euler_shader_info, nullptr, &euler_shader_module));
    std::cout << "Euler shader module loaded" << std::endl;

    // ============================================================================
    // DESCRIPTOR SET LAYOUT (2 Buffers: In/Out, Read-Only)
    // ============================================================================
    std::cout << "=== Setting up Euler Descriptor Sets ===" << std::endl;

    VkDescriptorSetLayoutBinding bindings_euler[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

    VkDescriptorSetLayoutCreateInfo layout_euler_info = {};
    layout_euler_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_euler_info.bindingCount = 2;
    layout_euler_info.pBindings = bindings_euler;

    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &layout_euler_info, nullptr, &resources.descriptorSetLayout));

    // ============================================================================
    // PIPELINE LAYOUT & PUSH CONSTANTS
    // ============================================================================
    VkPushConstantRange push_const_range_euler = {};
    push_const_range_euler.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_const_range_euler.offset = 0;
    push_const_range_euler.size = sizeof(EulerPushConstants);

    VkPipelineLayoutCreateInfo pipeline_layout_euler_info = {};
    pipeline_layout_euler_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_euler_info.setLayoutCount = 1;
    pipeline_layout_euler_info.pSetLayouts = &resources.descriptorSetLayout;
    pipeline_layout_euler_info.pushConstantRangeCount = 1;
    pipeline_layout_euler_info.pPushConstantRanges = &push_const_range_euler;

    VK_CHECK(vkCreatePipelineLayout(vk.device, &pipeline_layout_euler_info, nullptr, &resources.pipelineLayout));

    // ============================================================================
    // CREATE COMPUTE PIPELINE
    // ============================================================================
    VkPipelineShaderStageCreateInfo shader_stage_euler = {};
    shader_stage_euler.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_euler.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_euler.module = euler_shader_module;
    shader_stage_euler.pName = "main";

    VkComputePipelineCreateInfo compute_pipeline_euler_info = {};
    compute_pipeline_euler_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_euler_info.stage = shader_stage_euler;
    compute_pipeline_euler_info.layout = resources.pipelineLayout;

    VK_CHECK(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &compute_pipeline_euler_info, nullptr,
                                      &resources.pipeline));
    std::cout << "Euler compute pipeline created" << std::endl;

    vkDestroyShaderModule(vk.device, euler_shader_module, nullptr);

    // ============================================================================
    // CREATE DESCRIPTOR POOL & ALLOCATE SETS
    // ============================================================================
    VkDescriptorPoolSize pool_size_euler = {};
    pool_size_euler.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size_euler.descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info_euler = {};
    pool_info_euler.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info_euler.poolSizeCount = 1;
    pool_info_euler.pPoolSizes = &pool_size_euler;
    pool_info_euler.maxSets = 1;

    VK_CHECK(vkCreateDescriptorPool(vk.device, &pool_info_euler, nullptr, &resources.descriptorPool));

    VkDescriptorSetAllocateInfo alloc_info_euler = {};
    alloc_info_euler.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info_euler.descriptorPool = resources.descriptorPool;
    alloc_info_euler.descriptorSetCount = 1;
    alloc_info_euler.pSetLayouts = &resources.descriptorSetLayout;

    VK_CHECK(vkAllocateDescriptorSets(vk.device, &alloc_info_euler, &resources.descriptorSet));

    // ============================================================================
    // UPDATE DESCRIPTOR SETS WITH ACTUAL BUFFERS
    // ============================================================================
    VkDescriptorBufferInfo buffer_infos[2] = {{hidden_states.buffer, 0, hidden_states.size},
                                              {transformer_output.buffer, 0, transformer_output.size}};

    VkWriteDescriptorSet write_desc_euler[2];
    for (int i = 0; i < 2; i++)
    {
        write_desc_euler[i] = {};
        write_desc_euler[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_desc_euler[i].dstSet = resources.descriptorSet;
        write_desc_euler[i].dstBinding = i;
        write_desc_euler[i].dstArrayElement = 0;
        write_desc_euler[i].descriptorCount = 1;
        write_desc_euler[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write_desc_euler[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(vk.device, 2, write_desc_euler, 0, nullptr);
    std::cout << "Euler descriptor sets updated successfully" << std::endl;

    return resources;
}

#endif  // VK_HELPER_H
