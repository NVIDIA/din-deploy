// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "sam2_common.h"

namespace din::sam2
{

Sam2Config LoadSam2Config(const std::filesystem::path& metadata_path);
Image LoadPng(const std::filesystem::path& path);
void SaveRgbPng(const std::filesystem::path& path, const uint8_t* rgb, uint32_t width, uint32_t height);

Prompt LoadPrompt(const std::filesystem::path& path, uint32_t image_height, uint32_t image_width, int64_t image_size);
std::vector<std::filesystem::path> ListFrames(const std::filesystem::path& dir);
std::vector<std::filesystem::path> ResolveInputFrames(const std::filesystem::path& input, size_t max_frames = 0);
Prompt EmptyPropagatePrompt();
std::string FrameMaskName(size_t index, const char* suffix);

namespace detail
{

class AsyncMaskWriter
{
public:
    explicit AsyncMaskWriter(bool enabled);
    AsyncMaskWriter(const AsyncMaskWriter&) = delete;
    AsyncMaskWriter& operator=(const AsyncMaskWriter&) = delete;
    ~AsyncMaskWriter();

    void EnqueueRgb(const std::filesystem::path& path, const uint8_t* rgb, uint32_t width, uint32_t height);
    void Finish();

private:
    struct Job
    {
        std::filesystem::path path;
        std::vector<uint8_t> rgb;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    void ThrowIfWorkerFailed() const;
    void Run();

    static constexpr size_t kMaxQueuedJobs = 2;
    bool enabled_ = false;
    bool stop_ = false;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable space_cv_;
    std::deque<Job> jobs_;
    std::exception_ptr exception_;
};

}  // namespace detail

}  // namespace din::sam2
