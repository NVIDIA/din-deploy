// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <onnxruntime_cxx_api.h>

#if ORT_API_VERSION < 24
#error "ONNX Runtime headers are too old. TensorRT RTX EP registration requires ORT_API_VERSION >= 23."
#endif

#include "ort_session.h"

namespace
{

std::string ElementTypeName(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "uint64";
    default:
        return "unsupported";
    }
}

std::string ShapeString(const std::vector<int64_t>& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i != 0)
        {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

std::vector<int64_t> ConcreteShape(std::vector<int64_t> shape)
{
    if (shape.empty())
    {
        return {1};
    }

    for (auto& dim : shape)
    {
        if (dim <= 0)
        {
            dim = 1;
        }
    }

    return shape;
}

size_t ElementCount(const std::vector<int64_t>& shape)
{
    return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                               [](int64_t total, int64_t dim)
                                               {
                                                   return total * std::max<int64_t>(dim, 1);
                                               }));
}

struct InputBuffer
{
    std::vector<std::max_align_t> storage;
    std::vector<int64_t> shape;
};

template <typename T>
Ort::Value CreateZeroTensor(Ort::MemoryInfo& memory_info, InputBuffer& buffer)
{
    const size_t element_count = ElementCount(buffer.shape);
    const size_t bytes = element_count * sizeof(T);
    const size_t units = (bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    buffer.storage.assign(units, std::max_align_t{});
    return Ort::Value::CreateTensor<T>(memory_info, reinterpret_cast<T*>(buffer.storage.data()), element_count,
                                       buffer.shape.data(), buffer.shape.size());
}

Ort::Value CreateInputTensor(Ort::MemoryInfo& memory_info, ONNXTensorElementDataType type, InputBuffer& buffer)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return CreateZeroTensor<float>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return CreateZeroTensor<uint8_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return CreateZeroTensor<int8_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return CreateZeroTensor<uint16_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return CreateZeroTensor<int16_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return CreateZeroTensor<int32_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return CreateZeroTensor<int64_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return CreateZeroTensor<bool>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return CreateZeroTensor<double>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return CreateZeroTensor<uint32_t>(memory_info, buffer);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return CreateZeroTensor<uint64_t>(memory_info, buffer);
    default:
        throw std::runtime_error("Unsupported input element type: " + ElementTypeName(type));
    }
}

std::string OwnedNameToString(Ort::AllocatedStringPtr name)
{
    return name ? std::string{name.get()} : std::string{};
}

std::vector<std::pair<std::string, std::string>> ParseEpOptions(const std::string& options)
{
    std::vector<std::pair<std::string, std::string>> parsed_options;
    size_t option_start = 0;
    while (option_start < options.size())
    {
        const size_t option_end = options.find(';', option_start);
        const std::string option = options.substr(option_start, option_end - option_start);
        const size_t separator = option.find('=');
        if (separator == std::string::npos || separator == 0)
        {
            throw std::invalid_argument("--option entries must use key=value format, separated by semicolons.");
        }

        parsed_options.emplace_back(option.substr(0, separator), option.substr(separator + 1));
        if (option_end == std::string::npos)
        {
            break;
        }
        option_start = option_end + 1;
    }

    return parsed_options;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        argparse::ArgumentParser parser("din_base_onnx");
        parser.add_description("Load and run an ONNX model with the TensorRT RTX execution provider.");
        parser.add_argument("--model")
            .default_value(std::string{RESNET18_MODEL_PATH})
            .metavar("PATH")
            .help("ONNX model path.");
        parser.add_argument("--ep").default_value("trt-rtx").help("The execution provider to use 'trt-rtx' or 'cpu'.");
        parser.add_argument("--skip-compile").flag().help("Skip model compilation and directly load a model.");
        parser.add_argument("--option")
            .default_value(std::string{})
            .metavar("KEY=VALUE[;KEY=VALUE...]")
            .help("Additional execution-provider options.");
        parser.add_argument("--iterations")
            .default_value(static_cast<size_t>(1))
            .scan<'u', size_t>()
            .metavar("N")
            .help("Number of inference runs used for timing.");

        try
        {
            parser.parse_args(argc, argv);
        }
        catch (const std::exception& exception)
        {
            std::cerr << parser << '\n';
            throw std::runtime_error(exception.what());
        }

        const std::string model_path = parser.get<std::string>("--model");
        const std::string extra_ep_options = parser.get<std::string>("--option");
        const std::string ep = parser.get<std::string>("--ep");
        const bool skip_compile = parser.get<bool>("--skip-compile");
        const size_t iterations = parser.get<size_t>("--iterations");
        if (iterations == 0)
        {
            throw std::invalid_argument("--iterations must be greater than zero.");
        }

        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "din_base_onnx"};
        din::common::RegisterTensorRTRTXProvider(env);
        din::common::EpContextOptions ep_context;
        ep_context.output_dir = (std::filesystem::current_path() / "ep_cache").string();
        din::common::ModelProfile profile;
        if (skip_compile)
        {
            std::cout << "skipping compilation" << '\n';
            profile.skip_compile = skip_compile;
        }
        for (auto& option : ParseEpOptions(extra_ep_options))
        {
            profile.extra_ep_options.push_back(std::move(option));
        }
        profile.embed_ep_context = false;
        din::common::OrtRunner ort_runner(env, model_path, ep, "", ep_context, profile, nullptr);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const size_t input_count = ort_runner.session.GetInputCount();
        const size_t output_count = ort_runner.session.GetOutputCount();
        if (input_count == 0)
        {
            throw std::runtime_error("Model has no inputs.");
        }
        if (output_count == 0)
        {
            throw std::runtime_error("Model has no outputs.");
        }

        std::vector<std::string> input_names;
        std::vector<const char*> input_name_ptrs;
        std::vector<InputBuffer> input_buffers;
        std::vector<Ort::Value> input_tensors;

        input_names.reserve(input_count);
        input_name_ptrs.reserve(input_count);
        input_buffers.reserve(input_count);
        input_tensors.reserve(input_count);

        std::cout << "Model: " << model_path << '\n';
        std::cout << "Inputs:\n";

        for (size_t i = 0; i < input_count; ++i)
        {
            input_names.emplace_back(OwnedNameToString(ort_runner.session.GetInputNameAllocated(i, allocator)));
            input_name_ptrs.push_back(input_names.back().c_str());

            auto type_info = ort_runner.session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto element_type = tensor_info.GetElementType();

            InputBuffer buffer;
            buffer.shape = ConcreteShape(tensor_info.GetShape());

            std::cout << "  " << input_names.back() << " " << ElementTypeName(element_type) << " "
                      << ShapeString(buffer.shape) << '\n';

            input_buffers.push_back(std::move(buffer));
            input_tensors.push_back(CreateInputTensor(memory_info, element_type, input_buffers.back()));
        }

        std::vector<std::string> output_names;
        std::vector<const char*> output_name_ptrs;
        output_names.reserve(output_count);
        output_name_ptrs.reserve(output_count);

        std::cout << "Outputs:\n";
        for (size_t i = 0; i < output_count; ++i)
        {
            output_names.emplace_back(OwnedNameToString(ort_runner.session.GetOutputNameAllocated(i, allocator)));
            output_name_ptrs.push_back(output_names.back().c_str());
            std::cout << "  " << output_names.back() << '\n';
        }

        std::vector<double> run_times_ms;
        run_times_ms.reserve(iterations);
        std::vector<Ort::Value> outputs;
        for (size_t iteration = 0; iteration < iterations; ++iteration)
        {
            const auto start = std::chrono::steady_clock::now();
            outputs = ort_runner.session.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_tensors.data(),
                                             input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());
            const auto end = std::chrono::steady_clock::now();
            run_times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::sort(run_times_ms.begin(), run_times_ms.end());
        const size_t middle = run_times_ms.size() / 2;
        const double median_ms = run_times_ms.size() % 2 == 0 ? (run_times_ms[middle - 1] + run_times_ms[middle]) / 2.0
                                                              : run_times_ms[middle];

        std::cout << "Run completed. Produced " << outputs.size() << " output tensor(s).\n";
        std::cout << std::fixed << std::setprecision(3) << "Timing (" << iterations << " iteration"
                  << (iterations == 1 ? "" : "s") << "): "
                  << "min " << run_times_ms.front() << " ms, "
                  << "max " << run_times_ms.back() << " ms, "
                  << "median " << median_ms << " ms\n";
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            if (!outputs[i].IsTensor())
            {
                std::cout << "  " << output_names[i] << ": non-tensor output\n";
                continue;
            }

            const auto tensor_info = outputs[i].GetTensorTypeAndShapeInfo();
            std::cout << "  " << output_names[i] << " " << ElementTypeName(tensor_info.GetElementType()) << " "
                      << ShapeString(tensor_info.GetShape()) << '\n';
        }
    }
    catch (const Ort::Exception& exception)
    {
        std::cerr << "ONNX Runtime error: " << exception.what() << '\n';
        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
