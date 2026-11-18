// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "image.h"
#include "nvtx_helper.h"
#include "sam.h"
#include "sam2_io.h"
#include "sam3_pipeline.h"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

namespace
{
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;
    using json = nlohmann::json;

    double Seconds(Clock::time_point start)
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    // Both families measure the complete synchronous host adapter, including upload,
    // inference, output resize/download and completion.
    class FrameProcessor
    {
    public:
        FrameProcessor(din::sam::Sam& runtime, din::sam::PreparedPrompt prompt, size_t count)
            : runtime_(runtime)
              , state_(runtime.AcquireState())
              , initialization_{std::move(prompt), count, false}
              , validity_(runtime.MaxObjects())
        {
        }

        void Run(const din::io::RgbImageView& image, bool tracking)
        {
            probabilities_.resize(size_t(runtime_.MaxObjects()) * image.width * image.height);
            din::sam::FrameView frame;
            frame.data = image.rgb.data();
            frame.width = int(image.width);
            frame.height = int(image.height);
            frame.memory = din::sam::MemoryLocation::Host;
            din::sam::MaskOutputView output;
            output.data = probabilities_.data();
            output.width = frame.width;
            output.height = frame.height;
            output.memory = din::sam::MemoryLocation::Host;
            output.format = din::sam::MaskFormat::Probabilities;
            output.validity = validity_.data();
            const auto* previous = tracking && started_ ? state_.get() : nullptr;
            initialization_.tracking = tracking;
            auto submitted = runtime_.RunFrame(frame, output, *state_, previous, &initialization_);
            submitted.completion.Wait();
            started_ = tracking;
        }

        din::sam3::FrameResult Result(const din::io::RgbImageView& image) const
        {
            din::sam3::FrameResult result{image.width, image.height, {}};
            const size_t pixels = size_t(image.width) * image.height;
            for (size_t i = 0; i < validity_.size(); ++i)
            {
                if (validity_[i] <= 0)
                    continue;
                din::sam3::Instance instance;
                instance.id = uint32_t(i + 1);
                instance.probabilities.assign(probabilities_.begin() + i * pixels,
                                              probabilities_.begin() + (i + 1) * pixels);
                result.instances.push_back(std::move(instance));
            }
            return result;
        }

        size_t InstanceCount() const
        {
            return std::count_if(validity_.begin(), validity_.end(),
                                 [](float v)
                                 {
                                     return v > 0;
                                 });
        }

    private:
        din::sam::Sam& runtime_;
        din::sam::StateLease state_;
        din::sam::Initialization initialization_;
        std::vector<float> probabilities_, validity_;
        bool started_ = false;
    };
} // namespace

int main(int argc, char** argv)
{
    try
    {
        argparse::ArgumentParser parser("din_sam_cli");
        parser.add_description("SAM2 point/box and SAM3 text segmentation of PNG images or frame folders.");
        parser.add_argument("input").help("PNG image or lexically ordered PNG frame folder.");
        parser.add_argument("output").help("Directory for run.json and optional masks.");
        parser.add_argument("--model").required().choices("sam2", "sam3");
        parser.add_argument("--model-dir").required().help("Exported ONNX bundle directory.");
        parser.add_argument("--prompt").help("SAM2 prompt JSON in source-image coordinates.");
        parser.add_argument("--text").help("SAM3 text prompt.");
        parser.add_argument("--provider").default_value(std::string{"trt-rtx"}).choices("cpu", "trt-rtx");
        parser.add_argument("--ep-cache")
              .default_value(std::string{"artifacts/sam/runtime-cache"})
              .help("Shared cache root; family and graph/context caches are separated automatically.");
        parser.add_argument("--device").default_value(0).scan<'i', int>();
        parser.add_argument("--threshold")
              .default_value(.5f)
              .scan<'g', float>()
              .help("SAM3 detection confidence threshold; mask probabilities are thresholded at 0.5.");
        parser.add_argument("--tracking").default_value(std::string{"temporal"}).choices("temporal", "none").help(
            "temporal propagates recurrent state between frames (default, SAM3 tracking is experimental); "
            "none segments each frame independently.");
        parser.add_argument("--no-dump-masks")
              .default_value(false)
              .implicit_value(true)
              .help("Skip per-instance mask PNGs; run.json is still written.");
        parser.add_argument("--max-frames").default_value(0).scan<'i', int>();
        parser.parse_args(argc, argv);

        const std::string family = parser.get<std::string>("--model");
        const bool sam2 = family == "sam2";
        const bool track = parser.get<std::string>("--tracking") == "temporal";
        const bool dump = !parser.get<bool>("--no-dump-masks");
        const int maximum = parser.get<int>("--max-frames");
        const auto prompt_path = parser.present<std::string>("--prompt");
        const auto text = parser.present<std::string>("--text");
        if ((sam2 && (!prompt_path || text)) || (!sam2 && (!text || prompt_path)))
            throw std::invalid_argument("SAM2 requires --prompt and SAM3 requires --text; do not combine them");
        if (text && text->find_first_not_of(" \t\r\n") == std::string::npos)
            throw std::invalid_argument("SAM3 text prompt is empty");
        if (maximum < 0)
            throw std::invalid_argument("max-frames must be nonnegative");

        din::sam::SamConfig config;
        config.model = sam2 ? din::sam::ModelFamily::Sam2 : din::sam::ModelFamily::Sam3;
        config.model_dir = parser.get<std::string>("--model-dir");
        config.cache_dir = fs::path(parser.get<std::string>("--ep-cache")) / family;
        config.provider = parser.get<std::string>("--provider");
        config.device = parser.get<int>("--device");
        config.confidence_threshold = parser.get<float>("--threshold");
        if (config.device < 0 || !std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0 ||
            config.confidence_threshold > 1)
            throw std::invalid_argument("device must be nonnegative and threshold must be between 0 and 1");
        if (sam2 && parser.is_used("--threshold"))
            throw std::invalid_argument("--threshold requires --model sam3");

        const auto input_start = Clock::now();
        const fs::path input = parser.get<std::string>("input"), output = parser.get<std::string>("output");
        const auto frames = din::sam2::ResolveInputFrames(input, size_t(maximum));
        const auto [first_width, first_height] = din::io::ReadPngSize(frames.front());
        din::sam::Prompt prompt;
        if (sam2)
        {
            const auto metadata = din::sam2::LoadSam2Config(config.model_dir / "metadata.json");
            config.max_source_width = config.max_source_height = int(metadata.max_input_size);
            auto points = din::sam2::LoadPrompt(*prompt_path, first_height, first_width, metadata.image_size);
            if (points.labels.size() != size_t(metadata.max_points))
                throw std::invalid_argument("SAM2 prompt count does not match exported model");
            prompt.coordinates = std::move(points.coords);
            prompt.labels = std::move(points.labels);
        }
        else
        {
            config.max_source_width = config.max_source_height = 8192;
            prompt.text = *text;
        }
        if (first_width > uint32_t(config.max_source_width) || first_height > uint32_t(config.max_source_height))
            throw std::invalid_argument("Input exceeds the model's maximum source dimensions");
        fs::create_directories(output);
        const double input_setup = Seconds(input_start);

        auto started = Clock::now();
        din::sam::Sam runtime(config);
        const double session_load = Seconds(started);
        started = Clock::now();
        auto prepared = runtime.PreparePrompt(prompt);
        FrameProcessor processor(runtime, std::move(prepared), frames.size());
        const double prompt_setup = Seconds(started);

        std::vector<double> latency;
        double output_overhead = 0;
        auto records = json::array();
        // One owned result in flight bounds memory and overlaps PNG encoding with
        // the next frame. get() propagates write failures before run.json is saved.
        std::future<double> pending_write;
        size_t pending_frame = 0;
        const auto finish_write = [&]
        {
            if (pending_write.valid())
            {
                din::common::nvtx_scoped_range wait{"sam.cli.WaitForWriter"};
                records[pending_frame]["output_seconds"] =
                    records[pending_frame]["output_seconds"].get<double>() + pending_write.get();
            }
        };
        const auto processing_start = Clock::now();
        for (size_t frame = 0; frame < frames.size(); ++frame)
        {
            din::common::nvtx_scoped_range trace{"sam.cli.Frame"};
            started = Clock::now();
            const auto image = [&]
            {
                din::common::nvtx_scoped_range load{"sam.cli.LoadPng"};
                return din::io::LoadPng(frames[frame], runtime.HostInputBuffer());
            }();
            const double load_seconds = Seconds(started);
            if (image.width > uint32_t(config.max_source_width) || image.height > uint32_t(config.max_source_height))
                throw std::invalid_argument("Input exceeds the model's maximum source dimensions");
            started = Clock::now();
            processor.Run(image, track);
            const double inference_seconds = Seconds(started);
            latency.push_back(inference_seconds);
            const size_t instances = processor.InstanceCount();
            started = Clock::now();
            finish_write();
            const double output_wait = Seconds(started);
            started = Clock::now();
            if (dump)
            {
                din::common::nvtx_scoped_range snapshot{"sam.cli.SnapshotAndSubmitWriter"};
                pending_write = std::async(std::launch::async,
                                           [result = processor.Result(image), output, frame]
                                           {
                                               din::common::nvtx_scoped_range write{"sam.cli.WriteMasks"};
                                               const auto write_start = Clock::now();
                                               din::sam3::SaveResult(result, output, frame);
                                               return Seconds(write_start);
                                           });
                pending_frame = frame;
            }
            const double output_snapshot = Seconds(started);
            output_overhead += output_wait + output_snapshot;
            records.push_back({
                {"frame_index", frame},
                {"source", frames[frame].filename().string()},
                {"instances", instances},
                {"load_seconds", load_seconds},
                {"inference_seconds", inference_seconds},
                {"output_seconds", output_snapshot},
                {"output_wait_seconds", output_wait}
            });
            std::cout << "Frame " << frame + 1 << '/' << frames.size() << ": " << instances << " instances\n";
        }
        const double processing = Seconds(processing_start) - output_overhead;
        finish_write();
        const double wall = Seconds(processing_start);
        const double inference = std::accumulate(latency.begin(), latency.end(), 0.);
        const double mean = inference / latency.size();
        const double fps = processing > 0 ? frames.size() / processing : 0;
        json report{
            {"model", family},
            {"provider", config.provider},
            {"device", config.device},
            {"model_dir", fs::absolute(config.model_dir).string()},
            {"cache_dir", fs::absolute(config.cache_dir).string()},
            {
                "mode", track && frames.size() > 1
                            ? (sam2 ? "tracking" : "experimental_tracking")
                            : frames.size() > 1
                            ? "independent_frames"
                            : "image"
            },
            {"temporal_tracking", track},
            {"dump_masks", dump},
            {"input_setup_seconds", input_setup},
            {"session_load_seconds", session_load},
            {"prompt_setup_seconds", prompt_setup},
            {"initialization_seconds", session_load + prompt_setup},
            {"inference_seconds", latency},
            {"mean_inference_seconds", mean},
            {"processing_seconds", processing},
            {"processing_fps", fps},
            {"wall_seconds", wall},
            {"wall_fps", wall > 0 ? frames.size() / wall : 0},
            {"frames", records},
            {
                "timing_scope", "Host wall time; inference includes upload, mask resize/download and completion. "
                "Processing includes every PNG decode and frame I/O; excludes input setup, initialization, "
                "output snapshot/submission and writer waits. PNG writing overlaps processing; output_seconds "
                "includes snapshot/submission plus worker time, so frame times are not additive. "
                "Wall time includes writer waits and the final drain; excludes initialization and report writing."
            }
        };
        if (sam2)
            report["prompt"] = fs::absolute(*prompt_path).string();
        else
        {
            report["text"] = prompt.text;
            report["confidence_threshold"] = config.confidence_threshold;
        }
        std::ofstream report_file(output / "run.json");
        report_file.exceptions(std::ios::failbit | std::ios::badbit);
        report_file << report.dump(2) << '\n';
        std::cout << std::fixed << std::setprecision(3) << "Session load: " << session_load
            << " s\nPrompt setup: " << prompt_setup << " s\nMean inference: " << mean
            << " s\nProcessing: " << processing << " s (" << fps << " FPS)\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
