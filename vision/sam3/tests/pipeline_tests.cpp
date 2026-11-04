// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cuda_runtime_api.h>

namespace {
void Require(bool value, const char* message) {if (!value) throw std::runtime_error(message);}
template<class F> void Reject(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) {rejected = true;}
    Require(rejected, "Expected invalid result to be rejected");
}
void Check(cudaError_t status) {if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
void Compare(const din::sam3::FrameResult& a, const din::sam3::FrameResult& b) {
    Require(a.instances.size() == b.instances.size() && a.Labels() == b.Labels(), "Image/device/replay labels differ");
    for (size_t i = 0; i < a.instances.size(); ++i) {
        Require(a.instances[i].id == b.instances[i].id, "Replay IDs differ");
        for (size_t j = 0; j < a.instances[i].probabilities.size(); ++j)
            Require(std::abs(a.instances[i].probabilities[j] - b.instances[i].probabilities[j]) < 1e-5f, "Device probabilities differ");
    }
}
void CheckRasters(const din::sam3::Sam3Pipeline& pipeline, const din::sam3::FrameResult& result) {
    const auto* rasters = pipeline.LastRasterizedMasks();
    Require(rasters && rasters->labels == result.Labels() && rasters->combined == result.CombinedMask(), "CUDA raster labels/union differ");
    const size_t pixels = size_t(result.width) * result.height;
    Require(rasters->individual.size() == pixels * result.instances.size(), "CUDA raster size differs");
    for (size_t i = 0; i < result.instances.size(); ++i) {
        Require(rasters->ids[i] == result.instances[i].id, "CUDA raster IDs differ");
        for (size_t p = 0; p < pixels; ++p)
            Require(rasters->individual[i * pixels + p] == (result.instances[i].probabilities[p] > .5f ? 255 : 0), "CUDA individual mask differs");
    }
}
}
int main(int argc, char** argv) {
    try {
        using namespace din::sam3;
        FrameResult empty{3, 1, {}};
        Require(empty.Labels() == std::vector<uint32_t>({0, 0, 0}), "Empty result labels");
        FrameResult overlaps{3, 1, {{9, 1, {}, {.9f, .8f, .5f}}, {2, 1, {}, {.9f, .7f, .1f}}}};
        Require(overlaps.Labels() == std::vector<uint32_t>({2, 9, 0}), "Probability/ID tie ordering");
        Require(overlaps.CombinedMask() == std::vector<uint8_t>({255, 255, 0}), "Mask union");
        std::reverse(overlaps.instances.begin(), overlaps.instances.end());
        Require(overlaps.Labels() == std::vector<uint32_t>({2, 9, 0}), "Order-independent labels");
        auto invalid = overlaps; invalid.instances[0].id = 0; Reject([&] {invalid.Labels();});
        invalid = overlaps; invalid.instances[0].id = invalid.instances[1].id; Reject([&] {invalid.Labels();});
        invalid = overlaps; invalid.instances[0].probabilities[0] = std::numeric_limits<float>::quiet_NaN(); Reject([&] {invalid.Labels();});
        invalid = overlaps; invalid.instances[0].probabilities.clear(); Reject([&] {invalid.Labels();});
        invalid = empty; invalid.instances.resize(17); Reject([&] {invalid.Labels();});
        if (argc == 3) {
            Sam3PipelineConfig config; config.model_dir = argv[1];
            config.rasterize_output = true;
            config.confidence_threshold = .5f;
            Sam3Pipeline pipeline(config);
            const auto image = LoadPng(argv[2]);
            const auto expected = pipeline.SegmentImage(image, {"bicycle"});
            Require(expected.instances.size() == 3, "Bicycle fixture should contain three instances");
            CheckRasters(pipeline, expected);
            Compare(expected, pipeline.SegmentImage(image, {"bicycle"}));
            uint8_t* device = nullptr;
            size_t pitch = 0;
            cudaStream_t producer = nullptr;
            Check(cudaMallocPitch(reinterpret_cast<void**>(&device), &pitch, image.width * 3, image.height));
            try {
                Check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
                Check(cudaMemcpy2DAsync(device, pitch, image.rgb.data(), image.width * 3, image.width * 3, image.height, cudaMemcpyHostToDevice, producer));
                Compare(expected, pipeline.SegmentDeviceFrame({image.width, image.height, device, pitch, producer}, {"bicycle"}));
            } catch (...) {cudaStreamDestroy(producer); cudaFree(device); throw;}
            Check(cudaStreamDestroy(producer)); Check(cudaFree(device));
            Require(pipeline.SegmentImage(image, {"purple elephant"}).instances.empty(), "Absent concept must return no instances");
            Reject([&] {pipeline.ProcessNextFrame(image);});
            pipeline.ResetSequence({"bicycle"}, 2);
            const auto seed = pipeline.ProcessNextFrame(image);
            CheckRasters(pipeline, seed);
            const auto propagated = pipeline.ProcessNextFrame(image);
            CheckRasters(pipeline, propagated);
            Require(!seed.instances.empty() && !propagated.instances.empty(), "Tracking lost bicycle instances");
            Reject([&] {pipeline.ProcessNextFrame(image);});
            pipeline.ResetSequence({"bicycle"}, 2);
            Compare(seed, pipeline.ProcessNextFrame(image));
            Compare(propagated, pipeline.ProcessNextFrame(image));
            pipeline.ResetSequence({"purple elephant"}, 2);
            Require(pipeline.ProcessNextFrame(image).instances.empty(), "Empty video seed");
            const auto empty_frame = pipeline.ProcessNextFrame(image);
            Require(empty_frame.instances.empty(), "Empty video continuation");
            CheckRasters(pipeline, empty_frame);
            config.confidence_threshold = 0;
            Sam3Pipeline overflow(config);
            bool capacity_error = false;
            try {overflow.SegmentImage(image, {"bicycle"});}
            catch (const std::runtime_error& error) {capacity_error = std::string(error.what()).find("capacity exceeded") != std::string::npos;}
            Require(capacity_error, "Capacity overflow must be explicit");
        } else if (argc != 1) throw std::invalid_argument("Usage: pipeline_tests [model_dir bicycle.png]");
        std::cout << (argc == 3 ? "SAM3 result, device input, empty concept, overflow and image replay checks passed\n" : "SAM3 result contract checks passed\n");
    } catch (const std::exception& error) {std::cerr << error.what() << '\n'; return 1;}
}
