// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace din::sam3 {
struct Sam3PipelineConfig {
    std::filesystem::path model_dir = "D:/models/Sam3";
    std::filesystem::path cache_dir = "artifacts/sam3/runtime-cache";
    std::string provider = "trt-rtx";
    float confidence_threshold = 0.5f;
    bool profile = false;
    // Build compatibility export rasters from the returned host probabilities.
    bool rasterize_output = false;
    // Optional decoder diagnostics. Requires an instrumented graph exposing debug_* outputs.
    std::filesystem::path diagnostic_dir;
    std::filesystem::path diagnostic_decoder;
    std::filesystem::path diagnostic_graph_dir;
};
struct Image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgb;
};
struct TextPrompt { std::string text; };
// A borrowed, top-down RGB8 CUDA allocation on the pipeline's device. The caller
// retains the allocation until the device-frame method returns. producer_stream is a
// cudaStream_t (nullptr denotes the default stream); its prior work is awaited.
struct DeviceFrame {
    uint32_t width = 0, height = 0;
    const uint8_t* rgb = nullptr;
    size_t row_bytes = 0;
    void* producer_stream = nullptr;
};
struct Instance {
    uint32_t id = 0;
    float score = 0;
    std::array<float, 4> box{};
    std::vector<float> probabilities;
};
struct FrameResult {
    uint32_t width = 0, height = 0;
    std::vector<Instance> instances;
    std::vector<uint32_t> Labels() const;
    std::vector<uint8_t> CombinedMask() const;
};
struct RasterizedMasks {
    uint32_t width = 0, height = 0;
    std::vector<uint32_t> ids, labels;
    std::vector<uint8_t> combined, individual; // individual is instance-major
};
class Sam3Pipeline {
public:
    explicit Sam3Pipeline(Sam3PipelineConfig config);
    ~Sam3Pipeline();
    Sam3Pipeline(const Sam3Pipeline&) = delete;
    Sam3Pipeline& operator=(const Sam3Pipeline&) = delete;
    FrameResult SegmentImage(const Image& image, const TextPrompt& prompt);
    FrameResult SegmentDeviceFrame(const DeviceFrame& frame, const TextPrompt& prompt);
    // Experimental: tracks initial detections; association/reconditioning are incomplete.
    // The total clip length determines upstream pointer temporal embeddings.
    void ResetSequence(const TextPrompt& prompt, size_t frame_count);
    FrameResult ProcessNextFrame(const Image& image);
    FrameResult ProcessDeviceFrame(const DeviceFrame& frame);
    // Host enqueue times and deferred GPU event times; CPU readback waits are frame boundaries.
    const std::map<std::string, double>& LastTimings() const;
    // Valid until the next inference; null when GPU rasterization is disabled.
    const RasterizedMasks* LastRasterizedMasks() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
Image LoadPng(const std::filesystem::path& path);
void SaveResult(const FrameResult& result, const std::filesystem::path& directory, uint64_t frame_index,
                const RasterizedMasks* rasters = nullptr);
}
