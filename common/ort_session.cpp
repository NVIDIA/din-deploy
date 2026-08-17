// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ort_session.h"

#include <cuda.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "nvtx_helper.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// if the ep is not compiled in this repo we fallback to a binary contained in the app binary directory
#ifndef ONNXRUNTIME_TRT_RTX_EP_LIBRARY_PATH
#define ONNXRUNTIME_TRT_RTX_EP_LIBRARY_PATH ""
#endif

namespace din::common
{
namespace
{
namespace fs = std::filesystem;

fs::path ExecutableDirectory()
{
#ifdef _WIN32
    std::vector<wchar_t> executable_path(MAX_PATH);
    while (true)
    {
        const DWORD length =
            GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
        if (length == 0)
        {
            throw std::runtime_error("Failed to determine the executable path.");
        }
        if (static_cast<size_t>(length) < executable_path.size())
        {
            return fs::path{std::wstring{executable_path.data(), length}}.parent_path();
        }
        executable_path.resize(executable_path.size() * 2);
    }
#elif defined(__linux__)
    std::error_code error;
    const fs::path executable_path = fs::read_symlink("/proc/self/exe", error);
    if (error)
    {
        throw std::runtime_error("Failed to determine the executable path: " + error.message());
    }
    return executable_path.parent_path();
#else
#error "Determining the executable directory is not implemented for this platform."
#endif
}

class CudaDriverApi
{
public:
    CudaDriverApi()
    {
#ifdef _WIN32
        library_ = LoadLibraryW(L"nvcuda.dll");
        if (library_ == nullptr)
        {
            throw std::runtime_error("Failed to load nvcuda.dll");
        }
#else
        library_ = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (library_ == nullptr)
        {
            library_ = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        }
        if (library_ == nullptr)
        {
            throw std::runtime_error("Failed to load libcuda.so");
        }
#endif
        cuInit = LoadFunction<CuInitFn>("cuInit");
        cuDeviceGetCount = LoadFunction<CuDeviceGetCountFn>("cuDeviceGetCount");
        cuDeviceGet = LoadFunction<CuDeviceGetFn>("cuDeviceGet");
        cuDeviceGetAttribute = LoadFunction<CuDeviceGetAttributeFn>("cuDeviceGetAttribute");
        cuGetErrorName = LoadOptionalFunction<CuGetErrorNameFn>("cuGetErrorName");
#ifdef _WIN32
        cuDeviceGetLuid = LoadOptionalFunction<CuDeviceGetLuidFn>("cuDeviceGetLuid");
#endif
    }

    ~CudaDriverApi()
    {
#ifdef _WIN32
        if (library_ != nullptr)
        {
            FreeLibrary(library_);
        }
#else
        if (library_ != nullptr)
        {
            dlclose(library_);
        }
#endif
    }

    CudaDriverApi(const CudaDriverApi&) = delete;
    CudaDriverApi& operator=(const CudaDriverApi&) = delete;

    using CuInitFn = CUresult(CUDAAPI*)(unsigned int);
    using CuDeviceGetCountFn = CUresult(CUDAAPI*)(int*);
    using CuDeviceGetFn = CUresult(CUDAAPI*)(CUdevice*, int);
    using CuDeviceGetAttributeFn = CUresult(CUDAAPI*)(int*, CUdevice_attribute, CUdevice);
    using CuGetErrorNameFn = CUresult(CUDAAPI*)(CUresult, const char**);
#ifdef _WIN32
    using CuDeviceGetLuidFn = CUresult(CUDAAPI*)(char*, unsigned int*, CUdevice);
#endif

    CuInitFn cuInit = nullptr;
    CuDeviceGetCountFn cuDeviceGetCount = nullptr;
    CuDeviceGetFn cuDeviceGet = nullptr;
    CuDeviceGetAttributeFn cuDeviceGetAttribute = nullptr;
    CuGetErrorNameFn cuGetErrorName = nullptr;
#ifdef _WIN32
    CuDeviceGetLuidFn cuDeviceGetLuid = nullptr;
#endif

private:
    template <typename T>
    T LoadFunction(const char* name) const
    {
        T function = LoadOptionalFunction<T>(name);
        if (function == nullptr)
        {
            throw std::runtime_error(std::string("CUDA driver function not found: ") + name);
        }
        return function;
    }

    template <typename T>
    T LoadOptionalFunction(const char* name) const
    {
#ifdef _WIN32
        return reinterpret_cast<T>(GetProcAddress(library_, name));
#else
        return reinterpret_cast<T>(dlsym(library_, name));
#endif
    }

#ifdef _WIN32
    HMODULE library_ = nullptr;
#else
    void* library_ = nullptr;
#endif
};

CudaDriverApi& GetCudaDriverApi()
{
    static CudaDriverApi api;
    return api;
}

void ThrowOnCudaDriverError(CUresult result, const char* operation)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }

    std::string message = std::string(operation) + " failed";
    const auto& api = GetCudaDriverApi();
    if (api.cuGetErrorName != nullptr)
    {
        const char* error_name = nullptr;
        if (api.cuGetErrorName(result, &error_name) == CUDA_SUCCESS && error_name != nullptr)
        {
            message += ": ";
            message += error_name;
        }
    }
    throw std::runtime_error(message);
}

CudaDeviceGeneration GetCudaDeviceGeneration(int major, int minor)
{
    if (major < 8 || (major == 7 && minor <= 5))
    {
        return CudaDeviceGeneration::TuringOrOlder;
    }
    if (major == 8)
    {
        return CudaDeviceGeneration::AmpereOrAda;
    }
    if (major >= 10)
    {
        return CudaDeviceGeneration::BlackwellOrNewer;
    }
    return CudaDeviceGeneration::Other;
}

template <typename Metadata>
const std::string* FindMetadataValue(const Metadata& metadata, std::string_view key)
{
    const auto it = metadata.find(std::string(key));
    if (it == metadata.end())
    {
        return nullptr;
    }
    return &it->second;
}

int ParseCudaDeviceOverride(int device_count, const char* override_env_var)
{
    if (override_env_var == nullptr || override_env_var[0] == '\0')
    {
        return -1;
    }

    const char* value = std::getenv(override_env_var);
    if (value == nullptr || value[0] == '\0')
    {
        return -1;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed >= device_count)
    {
        throw std::runtime_error(std::string(override_env_var) + " must be a valid CUDA device ordinal");
    }

    return static_cast<int>(parsed);
}

std::optional<uint64_t> ParseUint64(std::string_view text)
{
    std::string copy(text);
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(copy.c_str(), &end, 0);
    if (end == copy.c_str() || *end != '\0')
    {
        return std::nullopt;
    }

    return static_cast<uint64_t>(parsed);
}

#ifdef _WIN32
uint64_t CudaLuidToUint64(const char (&luid)[8])
{
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i)
    {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(luid[i])) << (i * 8);
    }
    return value;
}
#else
struct PciBusId
{
    int domain = 0;
    int bus = 0;
    int device = 0;
    int function = 0;
};

int HexDigitValue(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return 10 + c - 'a';
    }
    if (c >= 'A' && c <= 'F')
    {
        return 10 + c - 'A';
    }
    return -1;
}

bool ParseHexUntil(std::string_view text, size_t& offset, char delimiter, int& value)
{
    int parsed = 0;
    const size_t start = offset;
    while (offset < text.size() && text[offset] != delimiter)
    {
        const int digit = HexDigitValue(text[offset]);
        if (digit < 0)
        {
            return false;
        }
        parsed = (parsed << 4) | digit;
        ++offset;
    }
    if (offset == start || offset >= text.size() || text[offset] != delimiter)
    {
        return false;
    }

    value = parsed;
    ++offset;
    return true;
}

bool ParseHexTail(std::string_view text, size_t& offset, int& value)
{
    int parsed = 0;
    const size_t start = offset;
    while (offset < text.size())
    {
        const int digit = HexDigitValue(text[offset]);
        if (digit < 0)
        {
            return false;
        }
        parsed = (parsed << 4) | digit;
        ++offset;
    }
    if (offset == start)
    {
        return false;
    }

    value = parsed;
    return true;
}

std::optional<PciBusId> ParsePciBusId(std::string_view text)
{
    PciBusId id{};
    size_t offset = 0;
    if (!ParseHexUntil(text, offset, ':', id.domain) || !ParseHexUntil(text, offset, ':', id.bus) ||
        !ParseHexUntil(text, offset, '.', id.device) || !ParseHexTail(text, offset, id.function))
    {
        return std::nullopt;
    }

    return id;
}
#endif

int GetCudaDeviceCount()
{
    auto& api = GetCudaDriverApi();
    ThrowOnCudaDriverError(api.cuInit(0), "cuInit");

    int device_count = 0;
    ThrowOnCudaDriverError(api.cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count <= 0)
    {
        throw std::runtime_error("No CUDA devices are available");
    }
    return device_count;
}

CUdevice GetCudaDevice(int cuda_device_ordinal)
{
    auto& api = GetCudaDriverApi();
    ThrowOnCudaDriverError(api.cuInit(0), "cuInit");
    CUdevice device{};
    ThrowOnCudaDriverError(api.cuDeviceGet(&device, cuda_device_ordinal), "cuDeviceGet");
    return device;
}

int GetCudaDeviceAttribute(CUdevice device, CUdevice_attribute attribute, const char* attribute_name)
{
    auto& api = GetCudaDriverApi();
    int value = 0;
    ThrowOnCudaDriverError(api.cuDeviceGetAttribute(&value, attribute, device), attribute_name);
    return value;
}

bool IsCudaUnifiedMemoryDevice(Ort::ConstEpDevice ep_device)
{
    const int cuda_device_ordinal = ChooseCudaDeviceOrdinal(ep_device, nullptr);
    const CUdevice cuda_device = GetCudaDevice(cuda_device_ordinal);
    // Integrated CUDA devices share physical memory with the CPU, so their
    // pinned host allocation is directly usable by the execution provider.
    return GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                  "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED)") != 0;
}

bool RegisterTensorRTRTXExecutionProvider(Ort::Env& env)
{
    static std::once_flag registration_once;
    static Ort::Env* registered_env = nullptr;
    std::call_once(registration_once,
                   [&env]
                   {
                       auto provider_library = std::filesystem::path{ONNXRUNTIME_TRT_RTX_EP_LIBRARY_PATH};
                       if (!std::filesystem::is_regular_file(provider_library))
                       {
#ifdef _WIN32
                           provider_library = ExecutableDirectory() / "onnxruntime_providers_nv_tensorrt_rtx.dll";
#else
                           provider_library = ExecutableDirectory() / "libonnxruntime_providers_nv_tensorrt_rtx.so";
#endif
                           if (!std::filesystem::is_regular_file(provider_library))
                           {
                               throw std::runtime_error("TensorRT RTX execution provider library not found: " +
                                                        provider_library.string());
                           }
                       }

#ifdef _WIN32
                       const auto provider_directory = provider_library.parent_path().wstring();
                       if (SetDllDirectoryW(provider_directory.c_str()) == 0)
                       {
                           throw std::runtime_error("Failed to add TensorRT RTX EP directory to the DLL search path.");
                       }
#endif

                       const auto provider_library_path = ToOrtPathString(provider_library);
                       env.RegisterExecutionProviderLibrary(kDinNvTensorRTRTXExecutionProvider,
                                                            provider_library_path.c_str());

                       const auto ep_devices = env.GetEpDevices();
                       std::cout << "Execution provider devices after TRT RTX registration:\n";
                       for (const auto& device : ep_devices)
                       {
                           std::cout << "  " << device.EpName() << " vendor=" << device.EpVendor()
                                     << " device_id=" << device.Device().DeviceId() << '\n';
                       }
                       registered_env = &env;
                   });

    if (registered_env != &env)
    {
        throw std::logic_error("TensorRT RTX was already registered on a different Ort::Env in this process.");
    }
    return true;
}

Ort::ConstEpDevice FindTensorRTRTXDevice(Ort::Env& env)
{
    std::vector<Ort::ConstEpDevice> selected_devices;
    const auto ep_devices = env.GetEpDevices();
    for (const auto& device : ep_devices)
    {
        if (std::string_view{device.EpName()} == kDinNvTensorRTRTXExecutionProvider)
        {
            selected_devices.push_back(device);
        }
    }

    if (selected_devices.empty())
    {
        throw std::runtime_error("No TensorRT RTX execution provider devices were found.");
    }
    return selected_devices.front();
}

void AppendTensorRTRTXEP(Ort::Env& env, Ort::SessionOptions& options, const std::string& cache_dir,
                         const std::string& model_path, const EpContextOptions& ep_context, const ModelProfile& profile,
                         Ort::SyncStream* compute_stream = nullptr)
{
    [[maybe_unused]] din::common::nvtx_scoped_range range{"append_trt_rtx_ep"};
    const auto device = FindTensorRTRTXDevice(env);
    if (!device)
    {
        throw std::runtime_error("TensorRT RTX EP device not found");
    }

    Ort::KeyValuePairs ep_options;
    if (profile.enable_cuda_graph)
    {
        ep_options.Add("enable_cuda_graph", "1");
    }

    const auto stem = fs::path(model_path).stem().string();
    const auto cache_path =
        (fs::path(cache_dir) / (profile.cache_subpath.empty() ? stem : profile.cache_subpath)).string();
    ep_options.Add("nv_runtime_cache_path", cache_path.c_str());
    if (!profile.min_shapes.empty())
    {
        ep_options.Add("nv_profile_min_shapes", profile.min_shapes.c_str());
        ep_options.Add("nv_profile_opt_shapes", profile.opt_shapes.c_str());
        ep_options.Add("nv_profile_max_shapes", profile.max_shapes.c_str());
    }

    if (compute_stream != nullptr)
    {
        const auto stream_address = std::to_string(reinterpret_cast<size_t>(compute_stream->GetHandle()));
        ep_options.Add("user_compute_stream", stream_address.c_str());
        ep_options.Add("has_user_compute_stream", "1");
    }
    for (const auto& [key, value] : profile.extra_ep_options)
    {
        ep_options.Add(key.c_str(), value.c_str());
    }

    const std::vector devices = {device};
    options.AppendExecutionProvider_V2(env, devices, ep_options);
}

std::string CompiledModelPath(const std::string& model_path, const EpContextOptions& ep_context,
                              const ModelProfile& profile)
{
    const fs::path input(model_path);
    const fs::path output_dir(ep_context.output_dir);
    const auto stem = input.stem().string();
    if (!profile.cache_subpath.empty())
    {
        return (output_dir / (stem + "." + profile.cache_subpath + ".embedded.onnx")).string();
    }
    return (output_dir / (stem + ".trt_rtx.embedded.onnx")).string();
}

bool IsCompatibleEpContext(Ort::Env& env, const fs::path& model_path)
{
    try
    {
        const auto device = FindTensorRTRTXDevice(env);
        const std::vector<Ort::ConstEpDevice> devices{device};
        const auto ort_model_path = ToOrtPathString(fs::absolute(model_path));
        Ort::AllocatorWithDefaultOptions allocator;
        const auto compatibility_info =
            Ort::GetCompatibilityInfoFromModelAllocated(ort_model_path.c_str(), device.EpName(), allocator);
        if (!compatibility_info)
        {
            std::cerr << "Warning: cached EP context has no compatibility information; recompiling " << model_path
                      << '\n';
            return false;
        }

        const auto compatibility = Ort::GetModelCompatibilityForEpDevices(devices, compatibility_info.get());
        if (compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL)
        {
            return true;
        }

        std::cerr << "Warning: cached EP context is not optimally compatible (status "
                  << static_cast<int>(compatibility) << "); recompiling " << model_path << '\n';
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "Warning: failed to validate cached EP context " << model_path << ": " << exception.what()
                  << "; recompiling\n";
    }
    return false;
}

std::string CompileEpContextModel(Ort::Env& env, const std::string& model_path, const std::string& cache_dir,
                                  const EpContextOptions& ep_context, const ModelProfile& profile)
{
    DIN_NVTX_FUNC_RANGE();
    const auto output_model_path = CompiledModelPath(model_path, ep_context, profile);
    if (fs::exists(output_model_path))
    {
        if (IsCompatibleEpContext(env, output_model_path))
        {
            return output_model_path;
        }

        // CompileModel writes a new context at this path. Remove both the ONNX
        // context and its possible external initializer sidecar first.
        fs::remove(output_model_path);
        fs::remove(output_model_path + ".data");
    }

    const auto input_model_abs = fs::absolute(model_path);
    const fs::path output_model(output_model_path);
    const auto output_dir = fs::absolute(output_model.parent_path());
    const auto output_model_abs = output_dir / output_model.filename();
    fs::create_directories(output_dir);

    Ort::SessionOptions compile_session_options;
    AppendTensorRTRTXEP(env, compile_session_options, cache_dir, model_path, ep_context, profile);

    Ort::ModelCompilationOptions compile_options(env, compile_session_options);
    compile_options.SetEpContextEmbedMode(profile.embed_ep_context);
    const auto input_wide = ToOrtPathString(input_model_abs);
    const auto output_wide = ToOrtPathString(output_model_abs);
    const auto output_external = output_model.filename().string() + ".data";
    const auto output_external_wide = ToOrtPathString(output_external);
    compile_options.SetInputModelPath(input_wide.c_str());
    compile_options.SetOutputModelPath(output_wide.c_str());
    const size_t size_threshold_external_init = 1024;
    compile_options.SetOutputModelExternalInitializersFile(output_external_wide.c_str(), size_threshold_external_init);

    const Ort::Status status = Ort::CompileModel(env, compile_options);
    if (!status.IsOK())
    {
        throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
    }
    return output_model_path;
}
}  // namespace

const char* ToString(CudaDeviceGeneration generation)
{
    switch (generation)
    {
    case CudaDeviceGeneration::TuringOrOlder:
        return "turing-or-older";
    case CudaDeviceGeneration::AmpereOrAda:
        return "ampere-or-ada";
    case CudaDeviceGeneration::BlackwellOrNewer:
        return "blackwell-or-newer";
    case CudaDeviceGeneration::Other:
        return "other";
    }
    return "unknown";
}

int ChooseCudaDeviceOrdinal(Ort::ConstEpDevice ep_device, const char* override_env_var)
{
    DIN_NVTX_FUNC_RANGE();
    const int device_count = GetCudaDeviceCount();

    const int override_ordinal = ParseCudaDeviceOverride(device_count, override_env_var);
    if (override_ordinal >= 0)
    {
        return override_ordinal;
    }

    const uint32_t ort_hardware_device_id = ep_device.Device().DeviceId();
    const auto metadata = ep_device.Device().Metadata().GetKeyValuePairs();
    std::vector<int> metadata_matches;

#ifdef _WIN32
    const std::string* luid_text = FindMetadataValue(metadata, "LUID");
    const auto ort_luid = luid_text != nullptr ? ParseUint64(*luid_text) : std::optional<uint64_t>{};
    if (luid_text != nullptr && !ort_luid.has_value())
    {
        std::cout << "Ignoring unparsable ORT LUID metadata value: " << *luid_text << std::endl;
    }
#else
    const std::string* pci_bus_id_text = FindMetadataValue(metadata, "pci_bus_id");
    const auto ort_pci_bus_id =
        pci_bus_id_text != nullptr ? ParsePciBusId(*pci_bus_id_text) : std::optional<PciBusId>{};
    if (pci_bus_id_text != nullptr && !ort_pci_bus_id.has_value())
    {
        std::cout << "Ignoring unparsable ORT pci_bus_id metadata value: " << *pci_bus_id_text << std::endl;
    }
#endif

    for (int ordinal = 0; ordinal < device_count; ++ordinal)
    {
        const CUdevice cuda_device = GetCudaDevice(ordinal);

#ifdef _WIN32
        if (ort_luid.has_value())
        {
            auto& api = GetCudaDriverApi();
            if (api.cuDeviceGetLuid == nullptr)
            {
                throw std::runtime_error("CUDA driver function not found: cuDeviceGetLuid");
            }

            char luid[8]{};
            unsigned int device_node_mask = 0;
            ThrowOnCudaDriverError(api.cuDeviceGetLuid(luid, &device_node_mask, cuda_device), "cuDeviceGetLuid");
            const uint64_t cuda_luid = CudaLuidToUint64(luid);
            const uint64_t cuda_luid_low = cuda_luid & 0xffffffffu;
            if (cuda_luid == *ort_luid || cuda_luid_low == *ort_luid)
            {
                metadata_matches.push_back(ordinal);
            }
        }
#else
        if (ort_pci_bus_id.has_value())
        {
            const int pci_domain_id = GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                                             "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)");
            const int pci_bus_id = GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                                          "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)");
            const int pci_device_id = GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                                             "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)");
            if (pci_domain_id == ort_pci_bus_id->domain && pci_bus_id == ort_pci_bus_id->bus &&
                pci_device_id == ort_pci_bus_id->device)
            {
                metadata_matches.push_back(ordinal);
            }
        }
#endif
    }

    if (metadata_matches.size() == 1)
    {
        return metadata_matches.front();
    }

    if (metadata_matches.size() > 1)
    {
        std::cout << "Multiple CUDA devices match ORT device metadata";
    }
    else
    {
        std::cout << "No CUDA device matches ORT device metadata";
    }
#ifdef _WIN32
    if (luid_text != nullptr)
    {
        std::cout << " LUID=" << *luid_text;
    }
#else
    if (pci_bus_id_text != nullptr)
    {
        std::cout << " pci_bus_id=" << *pci_bus_id_text;
    }
#endif
    std::cout << " for ORT hardware device_id=" << ort_hardware_device_id << "; defaulting to CUDA ordinal 0. Set "
              << (override_env_var != nullptr ? override_env_var : "FLUX_CUDA_DEVICE_ID") << " to override."
              << std::endl;

    return 0;
}

CudaGraphicsInteropSharedMemoryInfo QueryCudaGraphicsInteropSharedMemoryInfo(int cuda_device_ordinal,
                                                                             bool high_priority_compute_queue)
{
    DIN_NVTX_FUNC_RANGE();
    const CUdevice cuda_device = GetCudaDevice(cuda_device_ordinal);

    CudaGraphicsInteropSharedMemoryInfo info;
    info.cuda_device_ordinal = cuda_device_ordinal;
    info.compute_capability_major =
        GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)");
    info.compute_capability_minor =
        GetCudaDeviceAttribute(cuda_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)");

    info.generation = GetCudaDeviceGeneration(info.compute_capability_major, info.compute_capability_minor);
    info.supports_simultaneous_graphics_compute = info.compute_capability_major >= 8;

    if (!info.supports_simultaneous_graphics_compute)
    {
        info.max_shared_memory_bytes = 0;
    }
    else if (info.compute_capability_major >= 10 && high_priority_compute_queue)
    {
        info.max_shared_memory_bytes = 84 * 1024;
    }
    else
    {
        info.max_shared_memory_bytes = 48 * 1024;
    }

    return info;
}

// cppcheck-suppress constParameterReference
Ort::ConstEpDevice RegisterTensorRTRTXProvider(Ort::Env& env)
{
    DIN_NVTX_FUNC_RANGE();
    RegisterTensorRTRTXExecutionProvider(env);
    return FindTensorRTRTXDevice(env);
}

Ort::SyncStream CreateTensorRTRTXComputeStream(Ort::Env& env)
{
    const auto device = FindTensorRTRTXDevice(env);
    if (!device)
    {
        throw std::runtime_error("TensorRT RTX EP device not found");
    }
    auto stream = device.CreateSyncStream();
    if (stream.GetHandle() == nullptr)
    {
        throw std::runtime_error("ORT sync stream handle is null");
    }
    return stream;
}

OrtRunner::OrtRunner(Ort::Env& env_in, const std::string& model_path, const std::string& provider,
                     const std::string& cache_dir, const EpContextOptions& ep_context, const ModelProfile& profile,
                     Ort::SyncStream* compute_stream_in)
    : env(env_in)
{
    DIN_NVTX_FUNC_RANGE();
    auto session_model_path = model_path;

    if (provider == "trt-rtx")
    {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        options_.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        ep_device = FindTensorRTRTXDevice(env);
        if (!ep_device)
        {
            throw std::runtime_error("TensorRT RTX EP device not found");
        }
        uses_unified_memory_ = IsCudaUnifiedMemoryDevice(ep_device);
        if (compute_stream_in != nullptr)
        {
            compute_stream = compute_stream_in;
        }
        else
        {
            owned_compute_stream_ = ep_device.CreateSyncStream();
            compute_stream = &owned_compute_stream_;
        }

        if (!profile.skip_compile)
        {
            [[maybe_unused]] din::common::nvtx_scoped_range range{"load_or_compile_ep_context"};
            session_model_path = CompileEpContextModel(env, model_path, cache_dir, ep_context, profile);
        }
        AppendTensorRTRTXEP(env, options_, cache_dir, model_path, ep_context, profile, compute_stream);
    }
    else if (provider == "cpu")
    {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
    else
    {
        throw std::runtime_error("unsupported provider: " + provider);
    }

#ifdef _WIN32
    const auto wide = ToOrtPathString(session_model_path);
    {
        [[maybe_unused]] din::common::nvtx_scoped_range range{"create_ort_session"};
        session = Ort::Session(env, wide.c_str(), options_);
    }
#else
    {
        [[maybe_unused]] din::common::nvtx_scoped_range range{"create_ort_session"};
        session = Ort::Session(env, session_model_path.c_str(), options_);
    }
#endif
}

bool OrtRunner::HasDeviceIo() const
{
    return static_cast<bool>(ep_device) && compute_stream != nullptr && static_cast<bool>(*compute_stream);
}

bool OrtRunner::UsesUnifiedMemory() const
{
    return uses_unified_memory_;
}

Ort::UnownedAllocator OrtRunner::DeviceAllocator()
{
    const auto memory = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    if (!memory)
    {
        throw std::runtime_error("TensorRT RTX EP did not expose default device memory");
    }
    const auto allocator = env.GetSharedAllocator(memory);
    if (!allocator)
    {
        throw std::runtime_error("TensorRT RTX EP did not expose a default device allocator");
    }
    return allocator;
}

Ort::UnownedAllocator OrtRunner::PinnedAllocator()
{
    const auto memory = ep_device.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
    if (!memory)
    {
        throw std::runtime_error("TensorRT RTX EP did not expose pinned host memory");
    }
    const auto allocator = env.GetSharedAllocator(memory);
    if (!allocator)
    {
        throw std::runtime_error("TensorRT RTX EP did not expose a pinned host allocator");
    }
    return allocator;
}

Ort::ConstMemoryInfo OrtRunner::DeviceMemory()
{
    const auto memory = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    if (!memory)
    {
        throw std::runtime_error("TensorRT RTX EP did not expose default device memory");
    }
    return memory;
}

template <typename T>
TensorBuffer<T>::TensorBuffer(OrtRunner& runner, const std::vector<int64_t>& shape, bool use_device_io, bool disable_uma)
    : runner_(&runner)
    , use_device_io_(use_device_io)
    , use_unified_memory_(use_device_io && runner.UsesUnifiedMemory() && !disable_uma)
    , shape_(shape)
{
    if (use_device_io_)
    {
        host_tensor_ = Ort::Value::CreateTensor<T>(runner.PinnedAllocator(), shape_.data(), shape_.size());
        if (!use_unified_memory_)
        {
            device_tensor_ = Ort::Value::CreateTensor<T>(runner.DeviceAllocator(), shape_.data(), shape_.size());
        }
    }
    else
    {
        host_tensor_ = Ort::Value::CreateTensor<T>(cpu_allocator_, shape_.data(), shape_.size());
    }
}

template <typename T>
Ort::Value& TensorBuffer<T>::BindingValue()
{
    return use_device_io_ && !use_unified_memory_ ? device_tensor_ : host_tensor_;
}

template <typename T>
T* TensorBuffer<T>::HostData()
{
    return host_tensor_.GetTensorMutableData<T>();
}

template <typename T>
const T* TensorBuffer<T>::HostData() const
{
    return host_tensor_.GetTensorData<T>();
}

template <typename T>
void TensorBuffer<T>::Fill(T value)
{
    std::fill_n(HostData(), ElementCount(), value);
}

template <typename T>
void TensorBuffer<T>::CopyFromHostValue(Ort::Value& value)
{
    const auto count = value.GetTensorTypeAndShapeInfo().GetElementCount();
    std::copy_n(value.GetTensorData<T>(), count, HostData());
}

template <typename T>
void TensorBuffer<T>::CopyAsyncToDevice()
{
    if (use_device_io_ && !use_unified_memory_)
    {
        Copy(runner_->env, {&host_tensor_}, {&device_tensor_}, *runner_->compute_stream);
    }
}

template <typename T>
NotificationPtr TensorBuffer<T>::CopyAsyncToDeviceWithNotification()
{
    CopyAsyncToDevice();
    if (!use_device_io_)
    {
        return NotificationPtr{nullptr};
    }
    auto notification = NotificationPtr(*runner_->compute_stream);
    // Even when the transfer is a unified-memory no-op, record the stream so
    // callers retain a usable ordering point for Sync().
    notification.Record();
    return notification;
}

template <typename T>
void TensorBuffer<T>::CopyAsyncToHost()
{
    if (use_device_io_ && !use_unified_memory_)
    {
        Copy(runner_->env, {&device_tensor_}, {&host_tensor_}, *runner_->compute_stream);
    }
}

template <typename T>
NotificationPtr TensorBuffer<T>::CopyAsyncToHostWithNotification()
{
    CopyAsyncToHost();
    if (!use_device_io_)
    {
        return NotificationPtr{nullptr};
    }
    auto notification = NotificationPtr(*runner_->compute_stream);
    // A unified-memory buffer needs no copy, but the notification still waits
    // for device writes before its host data is consumed.
    notification.Record();
    return notification;
}

template <typename T>
void TensorBuffer<T>::CopyFrom(TensorBuffer& other)
{
    if (use_device_io_ != other.use_device_io_ || shape_ != other.shape_)
    {
        throw std::runtime_error("cannot copy incompatible tensor buffers");
    }
    if (use_device_io_ && !use_unified_memory_)
    {
        Copy(runner_->env, {&other.device_tensor_}, {&device_tensor_}, *runner_->compute_stream);
    }
    else
    {
        std::copy_n(other.HostData(), ElementCount(), HostData());
    }
}

template <typename T>
void TensorBuffer<T>::SwapWith(TensorBuffer& other)
{
    if (use_device_io_ != other.use_device_io_ || shape_ != other.shape_)
    {
        throw std::runtime_error("cannot swap incompatible tensor buffers");
    }
    std::swap(host_tensor_, other.host_tensor_);
    std::swap(device_tensor_, other.device_tensor_);
}

template <typename T>
void TensorBuffer<T>::Copy(Ort::Env& env, std::initializer_list<Ort::Value*> src,
                           std::initializer_list<Ort::Value*> dst, Ort::SyncStream& stream)
{
    std::vector<const OrtValue*> src_values;
    std::vector<OrtValue*> dst_values;
    src_values.reserve(src.size());
    dst_values.reserve(dst.size());
    std::transform(src.begin(), src.end(), std::back_inserter(src_values),
                   [](const auto* value)
                   {
                       return static_cast<const OrtValue*>(*value);
                   });
    std::transform(dst.begin(), dst.end(), std::back_inserter(dst_values),
                   [](const auto* value)
                   {
                       return static_cast<OrtValue*>(*value);
                   });
    Ort::ThrowOnError(Ort::GetApi().CopyTensors(env, src_values.data(), dst_values.data(), stream, src_values.size()));
}

template <typename T>
size_t TensorBuffer<T>::ElementCount() const
{
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                           [](size_t count, int64_t dim)
                           {
                               return count * static_cast<size_t>(dim);
                           });
}

// Explicitly instantiate TensorBuffer for every ONNXTensorElementDataType that
// has a standard C++ scalar supported by ORT's allocator-backed CreateTensor<T>.
// Entries without a standard scalar, or without support in this buffer path, are
// kept as comments so the ONNX enum coverage stays visible.
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: no tensor storage type
template class TensorBuffer<float>;     // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
template class TensorBuffer<uint8_t>;   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
template class TensorBuffer<int8_t>;    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
template class TensorBuffer<uint16_t>;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
template class TensorBuffer<int16_t>;   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
template class TensorBuffer<int32_t>;   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
template class TensorBuffer<int64_t>;   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
// template class TensorBuffer<std::string>;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: requires string-specific ORT APIs
template class TensorBuffer<bool>;            // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
template class TensorBuffer<Ort::Float16_t>;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
template class TensorBuffer<double>;          // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
template class TensorBuffer<uint32_t>;        // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
template class TensorBuffer<uint64_t>;        // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
// template class TensorBuffer<std::complex<float>>;   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: unsupported by this
// ORT CreateTensor<T> path template class TensorBuffer<std::complex<double>>;  //
// ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: unsupported by this ORT CreateTensor<T> path
template class TensorBuffer<Ort::BFloat16_t>;        // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
template class TensorBuffer<Ort::Float8E4M3FN_t>;    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN
template class TensorBuffer<Ort::Float8E4M3FNUZ_t>;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ
template class TensorBuffer<Ort::Float8E5M2_t>;      // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2
template class TensorBuffer<Ort::Float8E5M2FNUZ_t>;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4: packed 4-bit storage, no standard C++ scalar
// ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4: packed 4-bit storage, no standard C++ scalar
// ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1: packed 4-bit storage, no standard C++ scalar
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2: packed 2-bit storage, no standard C++ scalar
// ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2: packed 2-bit storage, no standard C++ scalar
// ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0
}  // namespace din::common
