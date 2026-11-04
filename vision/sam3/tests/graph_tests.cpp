// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "ort_session.h"
#include "sam3_artifact.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

std::vector<uint8_t> Read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Missing fixture: " + path.string());
    auto size = file.tellg();
    std::vector<uint8_t> result(static_cast<size_t>(size));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), size);
    if (!file) throw std::runtime_error("Cannot read fixture");
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("Usage: graph_tests model_dir graph_name cpu|trt-rtx");
        const std::filesystem::path directory = argv[1];
        const std::string name = argv[2], provider = argv[3];
        auto file = std::ifstream(directory / "metadata.json");
        const auto contract = nlohmann::json::parse(file).at("graphs").at(name);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam3_graph_test");
        if (provider == "trt-rtx") din::common::RegisterTensorRTRTXProvider(env);
        din::common::ModelProfile profile;
        profile.cache_subpath = name;
        profile.embed_ep_context = false;
        profile.enable_cuda_graph = false;
        std::string shapes;
        for (const auto& item : contract.at("inputs")) {
            if (!shapes.empty()) shapes += ',';
            shapes += item.at("name").get<std::string>() + ':';
            bool first = true;
            for (const auto& dimension : contract.at("fixture_inputs").at(item.at("name").get<std::string>())) {
                if (!first) shapes += 'x'; first = false;
                shapes += std::to_string(dimension.get<int64_t>());
            }
        }
        profile.min_shapes = profile.opt_shapes = profile.max_shapes = shapes;
        const auto cache = din::sam3::ArtifactCache(directory / "verification-cache", directory / (name + ".onnx"));
        din::common::OrtRunner runner(env, (directory / (name + ".onnx")).string(), provider,
                                      cache.string(), {(cache / "context").string()}, profile);
        Ort::IoBinding binding(runner.session);
        auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<std::vector<uint8_t>> storage;
        std::vector<Ort::Value> tensors;
        for (const auto& item : contract.at("inputs")) {
            const auto input_name = item.at("name").get<std::string>();
            const auto shape = contract.at("fixture_inputs").at(input_name).get<std::vector<int64_t>>();
            const auto dtype = item.at("dtype").get<std::string>();
            const std::map<std::string, ONNXTensorElementDataType> types{{"float", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
                {"int64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}, {"bool", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL}, {"uint8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8}};
            storage.push_back(Read(directory / (name + ".fixtures") / (input_name + ".bin")));
            tensors.push_back(Ort::Value::CreateTensor(cpu, storage.back().data(), storage.back().size(), shape.data(), shape.size(), types.at(dtype)));
            binding.BindInput(input_name.c_str(), tensors.back());
        }
        for (const auto& output : contract.at("outputs")) binding.BindOutput(output.at("name").get<std::string>().c_str(), cpu);
        runner.session.Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
        auto outputs = binding.GetOutputValues();
        auto names = binding.GetOutputNames();
        bool passed = true;
        auto report = nlohmann::json::object();
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto expected = Read(directory / (name + ".fixtures") / (names[i] + ".bin"));
            const auto info = outputs[i].GetTensorTypeAndShapeInfo();
            bool good = true;
            double maximum = 0, mean = 0;
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                if (expected.size() != info.GetElementCount() * sizeof(float)) throw std::runtime_error("Output size mismatch");
                const auto* actual = outputs[i].GetTensorData<float>();
                for (size_t j = 0; j < info.GetElementCount(); ++j) {
                    float reference;
                    std::memcpy(&reference, expected.data() + j * sizeof(float), sizeof(float));
                    const double error = std::abs(double(actual[j]) - reference);
                    maximum = std::max(maximum, error); mean += error;
                    good &= std::isfinite(actual[j]) && error <= .002 + .002 * std::abs(reference);
                }
                mean /= info.GetElementCount();
            } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                good = expected.size() == info.GetElementCount() && std::memcmp(expected.data(), outputs[i].GetTensorData<bool>(), expected.size()) == 0;
            } else throw std::runtime_error("Unsupported output dtype");
            report[names[i]] = {{"passed", good}, {"max_absolute_error", maximum}, {"mean_absolute_error", mean}};
            passed &= good;
        }
        std::cout << report.dump(2) << '\n';
        auto output = std::ofstream(directory / (name + "." + provider + ".parity.json"));
        output << report.dump(2);
        return passed ? 0 : 1;
    } catch (const std::exception& error) {std::cerr << error.what() << '\n'; return 1;}
}
