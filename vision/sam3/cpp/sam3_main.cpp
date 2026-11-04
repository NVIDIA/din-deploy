// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "sam3_pipeline.h"
#include "sam2_io.h"
#include "sam3_nvtx.h"
#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    try {
        din::sam3::Sam3PipelineConfig config;
        argparse::ArgumentParser parser("din_sam3_cli");
        parser.add_description("SAM3.1 text-prompted image segmentation and experimental causal tracking using exported ONNX graphs.");
        parser.add_argument("input"); parser.add_argument("output");
        parser.add_argument("--text").required();
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--provider").default_value(config.provider);
        parser.add_argument("--ep-cache").default_value(config.cache_dir.string());
        parser.add_argument("--diagnostic-dir").default_value(std::string{})
            .help("Write per-frame decoder debug outputs from an instrumented model (accuracy debugging only)");
        parser.add_argument("--diagnostic-decoder").default_value(std::string{})
            .help("Override the tracking decoder ONNX for accuracy diagnostics");
        parser.add_argument("--diagnostic-graph-dir").default_value(std::string{})
            .help("Override graphs present in this directory for controlled accuracy diagnostics");
        parser.add_argument("--threshold").default_value(.5f).scan<'g', float>();
        parser.add_argument("--warmup").default_value(0).scan<'i', int>();
        parser.add_argument("--repeat").default_value(1).scan<'i', int>();
        parser.add_argument("--framewise").default_value(false).implicit_value(true)
            .help("Segment a PNG frame folder independently; no temporal tracking or stable video IDs");
        parser.add_argument("--max-frames").default_value(0).scan<'i', int>();
        parser.add_argument("--profile").default_value(false).implicit_value(true).help("Record per-graph and tracking-stage wall times in run.json");
        parser.add_argument("--track").default_value(false).implicit_value(true)
            .help("Run causal multiplex tracking (experimental)");
        parser.parse_args(argc, argv);
        config.model_dir = parser.get<std::string>("--model-dir");
        config.provider = parser.get<std::string>("--provider");
        config.cache_dir = parser.get<std::string>("--ep-cache");
        config.confidence_threshold = parser.get<float>("--threshold");
        config.profile = parser.get<bool>("--profile");
        config.rasterize_output = true;
        config.diagnostic_dir = parser.get<std::string>("--diagnostic-dir");
        config.diagnostic_decoder = parser.get<std::string>("--diagnostic-decoder");
        config.diagnostic_graph_dir = parser.get<std::string>("--diagnostic-graph-dir");
        const int warmup = parser.get<int>("--warmup"), repeat = parser.get<int>("--repeat");
        if (warmup < 0 || repeat < 1) throw std::invalid_argument("Warmup must be nonnegative and repeat must be positive");
        const bool framewise = parser.get<bool>("--framewise");
        const bool track = parser.get<bool>("--track");
        if (track && (framewise || repeat != 1)) throw std::invalid_argument("Tracking requires repeat=1 and cannot use --framewise");
        const int maximum = parser.get<int>("--max-frames");
        if (maximum < 0) throw std::invalid_argument("max-frames must be nonnegative");
        const auto input_path = std::filesystem::path(parser.get<std::string>("input"));
        if (std::filesystem::is_directory(input_path) && !framewise && !track)
            throw std::invalid_argument("Frame folders require --track or --framewise");
        const auto frames = din::sam2::ResolveInputFrames(input_path, static_cast<size_t>(maximum));
        if (frames.empty()) throw std::invalid_argument("No input frames");
        const auto image = din::sam3::LoadPng(frames.front());
        auto started = std::chrono::steady_clock::now();
        din::sam3::Sam3Pipeline pipeline(config);
        const din::sam3::TextPrompt prompt{parser.get<std::string>("--text")};
        for (int i = 0; i < warmup; ++i) pipeline.SegmentImage(image, prompt);
        if (track) pipeline.ResetSequence(prompt, frames.size());
        const auto ready = std::chrono::steady_clock::now();
        din::sam3::FrameResult result;
        std::vector<double> latency;
        auto profiles = nlohmann::json::array();
        auto frame_records = nlohmann::json::array();
        for (size_t frame = 0; frame < frames.size(); ++frame) {
          const auto frame_label = "frame." + std::to_string(frame);
          din::sam3::TraceRange frame_trace{frame_label.c_str()};
          const auto current = frame == 0 ? image : din::sam3::LoadPng(frames[frame]);
          for (int i = 0; i < repeat; ++i) {

            const auto begin = std::chrono::steady_clock::now();
            result = track ? pipeline.ProcessNextFrame(current) : pipeline.SegmentImage(current, prompt);
            latency.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
            if (config.profile) profiles.push_back(pipeline.LastTimings());
          }
          din::sam3::SaveResult(result, parser.get<std::string>("output"), frame, pipeline.LastRasterizedMasks());
          frame_records.push_back({{"frame_index", frame}, {"source", frames[frame].filename().string()}, {"instances", result.instances.size()}});
          std::cout << "Frame " << frame + 1 << '/' << frames.size() << ": " << result.instances.size() << " instances\n" << std::flush;
        }
        const double initialization = std::chrono::duration<double>(ready - started).count();
        const double mean = std::accumulate(latency.begin(), latency.end(), 0.) / latency.size();
        nlohmann::json report{{"initialization_seconds", initialization}, {"inference_seconds", latency},
            {"stage_wall_ms", profiles},
            {"mode", track ? "experimental_tracking" : framewise ? "independent_frames" : "image"}, {"frames", frame_records}, {"temporal_tracking", track},
            {"mean_inference_seconds", mean}, {"warmup_iterations", warmup}, {"instances", result.instances.size()},
            {"provider", config.provider}, {"text", prompt.text}, {"model_dir", std::filesystem::absolute(config.model_dir).string()},
            {"diagnostic_graph_dir", config.diagnostic_graph_dir.empty() ? "" : std::filesystem::absolute(config.diagnostic_graph_dir).string()},
            {"diagnostic_decoder", config.diagnostic_decoder.empty() ? "" : std::filesystem::absolute(config.diagnostic_decoder).string()}};
        std::ofstream report_file(std::filesystem::path(parser.get<std::string>("output")) / "run.json");
        report_file.exceptions(std::ios::failbit | std::ios::badbit); report_file << report.dump(2);
        std::cout << "Instances: " << result.instances.size()
                  << "\nInitialization: " << initialization
                  << " s\nMean inference: " << mean << " s\n";
        return 0;
    } catch (const std::exception& error) {std::cerr << error.what() << '\n'; return 1;}
}
