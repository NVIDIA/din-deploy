// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
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
              , scores_(runtime.MaxObjects())
              , boxes_(runtime.MaxObjects() * 4)
        {
        }

        void Run(const din::io::RgbImage& image, bool tracking)
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
            output.scores = scores_.data();
            output.detection_boxes = boxes_.data();
            const auto* previous = tracking && started_ ? state_.get() : nullptr;
            initialization_.tracking = tracking;
            auto submitted = runtime_.RunFrame(frame, output, *state_, previous, &initialization_);
            submitted.completion.Wait();
            started_ = tracking;
        }

        din::sam3::FrameResult Result(const din::io::RgbImage& image, bool detector_boxes) const
        {
            din::sam3::FrameResult result{image.width, image.height, {}};
            const size_t pixels = size_t(image.width) * image.height;
            for (size_t i = 0; i < validity_.size(); ++i)
            {
                if (validity_[i] <= 0)
                    continue;
                din::sam3::Instance instance;
                instance.id = uint32_t(i + 1);
                instance.score = scores_[i];
                instance.probabilities.assign(probabilities_.begin() + i * pixels,
                                              probabilities_.begin() + (i + 1) * pixels);
                std::array<float, 4> box{float(image.width), float(image.height), 0, 0};
                for (size_t p = 0; p < pixels; ++p)
                {
                    const float value = instance.probabilities[p];
                    if (!std::isfinite(value) || value < 0 || value > 1)
                        throw std::runtime_error("SAM produced invalid probabilities");
                    if (value > .5f)
                    {
                        const float x = float(p % image.width), y = float(p / image.width);
                        box[0] = std::min(box[0], x);
                        box[1] = std::min(box[1], y);
                        box[2] = std::max(box[2], x + 1);
                        box[3] = std::max(box[3], y + 1);
                    }
                }
                if (detector_boxes)
                    std::copy_n(boxes_.data() + i * 4, 4, instance.box.data());
                else if (box[2] > 0)
                    instance.box = box;
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
        std::vector<float> probabilities_, validity_, scores_, boxes_;
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
        parser.add_argument("--track").default_value(false).implicit_value(true).help(
            "Propagate recurrent state (SAM3 tracking is experimental).");
        parser.add_argument("--framewise")
              .default_value(false)
              .implicit_value(true)
              .help("Segment each frame independently.");
        parser.add_argument("--dump-masks")
              .default_value(false)
              .implicit_value(true)
              .help("Write combined/instance PNGs, label NPYs and instance JSON for both families.");
        parser.add_argument("--max-frames").default_value(0).scan<'i', int>();
        parser.add_argument("--warmup").default_value(0).scan<'i', int>();
        parser.add_argument("--repeat").default_value(1).scan<'i', int>();
        parser.add_argument("--diagnostic-dir").default_value(std::string{});
        parser.add_argument("--diagnostic-decoder").default_value(std::string{});
        parser.add_argument("--diagnostic-graph-dir").default_value(std::string{});
        parser.parse_args(argc, argv);

        const std::string family = parser.get<std::string>("--model");
        const bool sam2 = family == "sam2";
        const bool track = parser.get<bool>("--track"), framewise = parser.get<bool>("--framewise");
        const bool dump = parser.get<bool>("--dump-masks");
        const int warmup = parser.get<int>("--warmup"), repeat = parser.get<int>("--repeat");
        const int maximum = parser.get<int>("--max-frames");
        const auto prompt_path = parser.present<std::string>("--prompt");
        const auto text = parser.present<std::string>("--text");
        if ((sam2 && (!prompt_path || text)) || (!sam2 && (!text || prompt_path)))
            throw std::invalid_argument("SAM2 requires --prompt and SAM3 requires --text; do not combine them");
        if (text && text->find_first_not_of(" \t\r\n") == std::string::npos)
            throw std::invalid_argument("SAM3 text prompt is empty");
        if (warmup < 0 || repeat < 1 || maximum < 0)
            throw std::invalid_argument("warmup/max-frames must be nonnegative and repeat must be positive");
        if (track && (framewise || repeat != 1))
            throw std::invalid_argument("Tracking requires repeat=1 and cannot use --framewise");

        din::sam::SamConfig config;
        config.model = sam2 ? din::sam::ModelFamily::Sam2 : din::sam::ModelFamily::Sam3;
        config.model_dir = parser.get<std::string>("--model-dir");
        config.cache_dir = fs::path(parser.get<std::string>("--ep-cache")) / family;
        config.provider = parser.get<std::string>("--provider");
        config.device = parser.get<int>("--device");
        config.confidence_threshold = parser.get<float>("--threshold");
        config.diagnostic_dir = parser.get<std::string>("--diagnostic-dir");
        config.diagnostic_decoder = parser.get<std::string>("--diagnostic-decoder");
        config.diagnostic_graph_dir = parser.get<std::string>("--diagnostic-graph-dir");
        if (config.device < 0 || !std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0 ||
            config.confidence_threshold > 1)
            throw std::invalid_argument("device must be nonnegative and threshold must be between 0 and 1");
        if (sam2 && (parser.is_used("--threshold") || !config.diagnostic_dir.empty() ||
            !config.diagnostic_decoder.empty() || !config.diagnostic_graph_dir.empty()))
            throw std::invalid_argument("threshold and diagnostic options require --model sam3");

        const auto input_start = Clock::now();
        const fs::path input = parser.get<std::string>("input"), output = parser.get<std::string>("output");
        if (fs::is_directory(input) && !track && !framewise)
            throw std::invalid_argument("Frame folders require --track or --framewise");
        const auto frames = din::sam2::ResolveInputFrames(input, size_t(maximum));
        const auto first = din::io::LoadPng(frames.front());
        din::sam::Prompt prompt;
        if (sam2)
        {
            const auto metadata = din::sam2::LoadSam2Config(config.model_dir / "metadata.json");
            config.max_source_width = config.max_source_height = int(metadata.max_input_size);
            auto points = din::sam2::LoadPrompt(*prompt_path, first.height, first.width, metadata.image_size);
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
        if (first.width > uint32_t(config.max_source_width) || first.height > uint32_t(config.max_source_height))
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
        started = Clock::now();
        for (int i = 0; i < warmup; ++i)
            processor.Run(first, false);
        const double warmup_seconds = Seconds(started);

        std::vector<double> latency;
        auto records = json::array();
        const auto processing_start = Clock::now();
        for (size_t frame = 0; frame < frames.size(); ++frame)
        {
            din::common::nvtx_scoped_range trace{"sam.cli.Frame"};
            started = Clock::now();
            const auto image = frame == 0 ? first : din::io::LoadPng(frames[frame]);
            const double load_seconds = Seconds(started);
            if (image.width > uint32_t(config.max_source_width) || image.height > uint32_t(config.max_source_height))
                throw std::invalid_argument("Input exceeds the model's maximum source dimensions");
            std::vector<double> frame_latency;
            for (int i = 0; i < repeat; ++i)
            {
                started = Clock::now();
                processor.Run(image, track);
                const double elapsed = Seconds(started);
                frame_latency.push_back(elapsed);
                latency.push_back(elapsed);
            }
            const size_t instances = processor.InstanceCount();
            started = Clock::now();
            if (dump)
                din::sam3::SaveResult(processor.Result(image, !sam2 && !track), output, frame);
            const double output_seconds = Seconds(started);
            records.push_back({
                {"frame_index", frame},
                {"source", frames[frame].filename().string()},
                {"instances", instances},
                {"load_seconds", load_seconds},
                {"inference_seconds", frame_latency},
                {"output_seconds", output_seconds}
            });
            std::cout << "Frame " << frame + 1 << '/' << frames.size() << ": " << instances << " instances\n";
        }
        const double processing = Seconds(processing_start);
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
                "mode", track
                            ? (sam2 ? "tracking" : "experimental_tracking")
                            : framewise
                            ? "independent_frames"
                            : "image"
            },
            {"temporal_tracking", track},
            {"dump_masks", dump},
            {"input_setup_seconds", input_setup},
            {"session_load_seconds", session_load},
            {"prompt_setup_seconds", prompt_setup},
            {"initialization_seconds", session_load + prompt_setup},
            {"warmup_iterations", warmup},
            {"warmup_seconds", warmup_seconds},
            {"repeat", repeat},
            {"inference_seconds", latency},
            {"mean_inference_seconds", mean},
            {"processing_seconds", processing},
            {"processing_fps", fps},
            {"frames", records},
            {
                "timing_scope", "Host wall time; inference includes upload, mask resize/download and completion. "
                "Processing includes frame I/O and repeats; excludes input setup (including first "
                "PNG decode), initialization, warmup and report writing."
            },
            {"diagnostic_graph_dir", config.diagnostic_graph_dir.string()},
            {"diagnostic_decoder", config.diagnostic_decoder.string()}
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
            << " s\nPrompt setup: " << prompt_setup << " s\nWarmup: " << warmup_seconds
            << " s\nMean inference: " << mean << " s\nProcessing: " << processing << " s (" << fps << " FPS)\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
