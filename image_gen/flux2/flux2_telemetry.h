// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <utility>

#include "flux2_cli.h"

// Stage timings are synchronized wall times, including stage transfers.
// Synchronization is supplied by each backend to respect its ORT/graphics ordering.
class Flux2Telemetry
{
public:
    Flux2Telemetry(int steps, Flux2Progress progress, std::function<void()> synchronize)
        : steps_(steps)
          , progress_(std::move(progress))
          , synchronize_(std::move(synchronize))
          , start_(Clock::now())
          , stage_start_(start_)
    {
        Report("rng", 0);
    }

    void Begin(const char* stage, double Flux2Timings::* field)
    {
        synchronize_();
        const auto now = Clock::now();
        timings_.*field_ += Milliseconds(now - stage_start_);
        stage_start_ = now;
        field_ = field;
        Report(stage, 0);
    }

    void Step(int completed)
    {
        if (progress_)
        {
            synchronize_();
            Report("denoise", completed);
        }
    }

    Flux2Timings Finish()
    {
        synchronize_();
        const auto now = Clock::now();
        timings_.*field_ += Milliseconds(now - stage_start_);
        timings_.total_ms = Milliseconds(now - start_);
        timings_.per_step_ms = timings_.denoise_ms / steps_;
        Report("done", steps_);
        return timings_;
    }

private:
    using Clock = std::chrono::steady_clock;

    static double Milliseconds(Clock::duration duration)
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    void Report(const char* stage, int completed)
    {
        if (progress_)
            progress_(stage, completed, steps_);
    }

    int steps_;
    Flux2Progress progress_;
    std::function<void()> synchronize_;
    Clock::time_point start_, stage_start_;
    Flux2Timings timings_;
    double Flux2Timings::* field_ = &Flux2Timings::rng_ms;
};
