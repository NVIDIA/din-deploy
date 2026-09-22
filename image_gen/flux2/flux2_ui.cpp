// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <thread>

#include "flux2.h"
#include "flux2_cli.h"
#include "io/image.h"
#include <argparse/argparse.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace
{
    // Stable per-root namespace prevents a different model directory from reusing stale engines.
    std::string CacheNamespace(const std::filesystem::path& root)
    {
        const auto path = std::filesystem::weakly_canonical(root).generic_u8string();
        uint64_t hash = 14695981039346656037ull;
        for (const auto c : path)
        {
            hash ^= static_cast<unsigned char>(c);
            hash *= 1099511628211ull;
        }
        return "root_" + std::to_string(hash);
    }

    struct Request
    {
        Flux2Config config;
        Flux2GenerationOptions generation;
        bool random_seeds = false;
    };

    struct Result
    {
        Flux2Image image;
        unsigned int seed = 0;
        GLuint texture = 0;
    };

    // Only the worker touches the pipeline; all graphics resources belong to the UI thread.
    class Generator
    {
    public:
        Generator()
            : worker_(
                [this]
                {
                    Run();
                })
        {
        }

        ~Generator()
        {
            {
                std::lock_guard lock(mutex);
                closing_ = true;
            }
            cv_.notify_one();
            worker_.join();
        }

        void Submit(Request request)
        {
            std::lock_guard lock(mutex);
            if (busy)
                return;
            busy = true;
            failed = false;
            status = "Loading models (first compilation can take several minutes)";
            completed = 0;
            total = request.config.steps * static_cast<int>(request.config.num_images);
            request_ = std::move(request);
            cv_.notify_one();
        }

        std::mutex mutex;
        bool busy = false, failed = false;
        std::string status = "Ready";
        int completed = 0, total = 1;
        size_t used_mb = 0, total_mb = 0;
        std::vector<Result> ready;

    private:
        static bool SameModels(const Flux2Config& a, const Flux2Config& b)
        {
            return a.model_dir == b.model_dir && a.processing == b.processing && a.provider == b.provider &&
                a.precision == b.precision && a.text_encoder == b.text_encoder && a.steps == b.steps &&
                a.weight_streaming_budget == b.weight_streaming_budget &&
                a.ep_cache_dir == b.ep_cache_dir && a.ep_context_dir == b.ep_context_dir;
        }

        void UpdateMemory()
        {
            size_t free = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free, &total_bytes) == cudaSuccess)
            {
                std::lock_guard lock(mutex);
                used_mb = (total_bytes - free) / (1024 * 1024);
                total_mb = total_bytes / (1024 * 1024);
            }
            else
                cudaGetLastError();
        }

        void Run()
        {
            std::unique_ptr<Flux2ProcessingPipeline> pipeline;
            std::optional<Flux2Config> loaded;
            std::mt19937 random(std::random_device{}());
            for (;;)
            {
                std::unique_lock lock(mutex);
                cv_.wait_for(lock, std::chrono::milliseconds(500),
                             [&]
                             {
                                 return closing_ || request_.has_value();
                             });
                if (closing_)
                    break;
                if (!request_)
                {
                    lock.unlock();
                    if (loaded && loaded->provider == Flux2ExecutionProvider::TrtRtx)
                        UpdateMemory();
                    continue;
                }
                Request request = std::move(*request_);
                request_.reset();
                lock.unlock();
                try
                {
                    const auto& config = request.config;
                    ValidateFlux2Config(config);
                    if (!loaded || !SameModels(*loaded, config))
                    {
                        pipeline.reset(); // Release old engines before allocating the replacement.
                        loaded.reset();
                        pipeline = CreateFlux2Pipeline(config);
                        pipeline->Initialize();
                        loaded = config;
                    }
                    pipeline->SetPrompt(config.prompt);
                    for (unsigned int i = 0; i < config.num_images; ++i)
                    {
                        const unsigned int seed = request.random_seeds ? random() : config.seed + i;
                        auto image = pipeline->GenerateImage(
                            seed,
                            [&](const char* stage, int step, int)
                            {
                                {
                                    std::lock_guard guard(mutex);
                                    completed = static_cast<int>(i) * config.steps + step;
                                    status = "Image " + std::to_string(i + 1) + "/" + std::to_string(config.num_images)
                                        +
                                        ": " + stage;
                                }
                                if (config.provider == Flux2ExecutionProvider::TrtRtx)
                                    UpdateMemory();
                            },
                            request.generation);
                        if (!config.output_path.empty())
                        {
                            std::filesystem::create_directories(config.output_path);
                            const auto file =
                                config.output_path / ("flux2_" + std::to_string(i) + "_" + std::to_string(seed) +
                                    ".png");
                            if (!din::io::SaveRgbFloatImage(file.string(), image.data.data(), image.height, image.width,
                                                            din::io::ImageValueRange::MinusOneToOne))
                                throw std::runtime_error("Could not save " + file.string());
                        }
                        std::lock_guard guard(mutex);
                        ready.push_back({std::move(image), seed, 0});
                    }
                    std::lock_guard guard(mutex);
                    completed = total;
                    status = "Done";
                }
                catch (const std::exception& error)
                {
                    pipeline.reset();
                    loaded.reset();
                    std::lock_guard guard(mutex);
                    status = std::string("Error: ") + error.what();
                    failed = true;
                }
                std::lock_guard guard(mutex);
                busy = false;
            }
            // Destroy sessions, streams and GPU allocations on their owning thread.
        }

        std::condition_variable cv_;
        bool closing_ = false;
        std::optional<Request> request_;
        std::thread worker_;
    };

    void Upload(Result& result)
    {
        const auto& img = result.image;
        const size_t pixels = static_cast<size_t>(img.width) * img.height;
        if (!pixels || img.data.size() != pixels * 3)
            throw std::runtime_error("Invalid image returned by pipeline");
        std::vector<unsigned char> rgb(pixels * 3);
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c)
            {
                const float v = img.data[c * pixels + p];
                rgb[p * 3 + c] =
                    std::isfinite(v) ? static_cast<unsigned char>(std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f) : 0;
            }
        glGenTextures(1, &result.texture);
        glBindTexture(GL_TEXTURE_2D, result.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    }

    int Studio(const Flux2Config& initial, bool smoke, bool auto_generate, bool exit_after)
    {
        Generator generator;
        Flux2Config config = initial;
        std::array<char, 2048> model{}, output{};
        std::array<char, 8192> prompt{};
        std::snprintf(model.data(), model.size(), "%s", config.model_dir.string().c_str());
        std::snprintf(output.data(), output.size(), "%s", config.output_path.string().c_str());
        std::snprintf(prompt.data(), prompt.size(), "%s", config.prompt.c_str());
        int streaming_mode = 0; // Auto / Manual / Off; independent of session configuration.
        int budget = 50;
        int count = static_cast<int>(config.num_images);
        int encoder = config.text_encoder == Flux2TextEncoder::Qwen3_4B ? 0 : 1;
        int precision = config.precision == "bf16"
                            ? 0
                            : config.precision == "fp16"
                            ? 1
                            : config.precision == "fp8"
                            ? 2
                            : 3;
        const char* precisions[] = {"bf16", "fp16", "fp8", "nvfp4"};
        const char* backends[] = {"CPU", "CUDA", "DirectX", "DirectX CIG", "Vulkan", "Vulkan CIG"};
        int backend = static_cast<int>(config.processing);
        int provider = config.provider == Flux2ExecutionProvider::Cpu ? 0 : 1;
        std::vector<Result> results;
        int selected = -1, frames = 0;
        bool launched = false;
        int exit_code = 0;
        GLFWwindow* window = glfwGetCurrentContext();
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos({0, 0});
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("Flux2 Studio", nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            ImGui::TextUnformatted("DIN Deploy - FLUX.2-klein Interactive Studio");
            ImGui::Separator();
            bool busy, failed;
            int completed, total;
            std::string status;
            size_t used_mb, total_mb;
            {
                std::lock_guard lock(generator.mutex);
                busy = generator.busy;
                failed = generator.failed;
                status = generator.status;
                completed = generator.completed;
                total = generator.total;
                used_mb = generator.used_mb;
                total_mb = generator.total_mb;
                for (auto& result : generator.ready)
                {
                    results.push_back(std::move(result));
                    selected = static_cast<int>(results.size()) - 1;
                }
                generator.ready.clear();
            }
            if (launched && !busy && exit_after)
            {
                exit_code = failed ? 1 : 0;
                glfwSetWindowShouldClose(window, true);
            }
            ImGui::BeginChild("controls", ImVec2(430, 0), true);
            ImGui::BeginDisabled(busy);
            ImGui::InputText("Model root", model.data(), model.size());
            ImGui::InputText("Save directory", output.data(), output.size());
            ImGui::TextUnformatted("Prompt");
            ImGui::InputTextMultiline("##prompt", prompt.data(), prompt.size(), ImVec2(-1, 100));
            ImGui::RadioButton("Qwen3-4B", &encoder, 0);
            const bool translator_exists = std::filesystem::is_regular_file(std::filesystem::path(model.data()) /
                "text_encoder_translator/model.onnx");
            ImGui::BeginDisabled(!translator_exists);
            ImGui::RadioButton("Qwen3-0.6B + translator", &encoder, 1);
            ImGui::EndDisabled();
            if (!translator_exists)
                ImGui::TextWrapped("Translator ONNX is missing from this model root.");
            ImGui::Combo("Precision", &precision, precisions, IM_ARRAYSIZE(precisions));
            ImGui::Combo("Provider", &provider, "CPU\0TensorRT RTX\0");
            if (provider == 0)
            {
                backend = 0;
                streaming_mode = 0;
            }
            if (ImGui::BeginCombo("Processing", backends[backend]))
            {
                for (int i = 0; i < 6; ++i)
                {
                    if (!IsFlux2BackendAvailable(static_cast<Flux2ProcessingBackend>(i)) || (provider == 0 && i != 0))
                        continue;
                    if (ImGui::Selectable(backends[i], backend == i))
                        backend = i;
                }
                ImGui::EndCombo();
            }
            ImGui::BeginDisabled(provider == 0);
            ImGui::Combo("Weight streaming", &streaming_mode, "Auto\0Manual\0Off\0");
            if (streaming_mode == 1)
                ImGui::SliderInt("Resident weights %", &budget, 10, 100);
            ImGui::EndDisabled();
            ImGui::InputScalar("Seed", ImGuiDataType_U32, &config.seed);
            ImGui::SliderInt("Denoise steps", &config.steps, 1, 50);
            ImGui::SliderInt("Images", &count, 1, 16);
            bool generate = ImGui::Button("Generate", ImVec2(-1, 35));
            bool random = ImGui::Button("Random sweep", ImVec2(-1, 30));
            ImGui::EndDisabled();
            if (!busy && (generate || random || (auto_generate && !launched)))
            {
                for (auto& result : results)
                    if (result.texture)
                        glDeleteTextures(1, &result.texture);
                results.clear();
                selected = -1;
                config.model_dir = model.data();
                config.output_path = output.data();
                config.prompt = prompt.data();
                const auto cache_namespace = CacheNamespace(config.model_dir);
                config.ep_cache_dir = initial.ep_cache_dir / cache_namespace;
                config.ep_context_dir = initial.ep_context_dir / cache_namespace;
                config.text_encoder = encoder == 0 ? Flux2TextEncoder::Qwen3_4B : Flux2TextEncoder::Qwen3_06BTranslator;
                config.precision = precisions[precision];
                config.num_images = static_cast<unsigned int>(count);
                config.processing = static_cast<Flux2ProcessingBackend>(backend);
                config.provider = provider == 0 ? Flux2ExecutionProvider::Cpu : Flux2ExecutionProvider::TrtRtx;
                config.weight_streaming_budget = provider == 0 ? "" : "-1";
                Flux2GenerationOptions generation;
                if (provider != 0 && streaming_mode != 0)
                    generation.weight_streaming_budget = streaming_mode == 1 ? std::to_string(budget) + "%" : "0";
                generator.Submit({config, generation, random});
                launched = true;
            }
            ImGui::ProgressBar(static_cast<float>(completed) / std::max(total, 1), ImVec2(-1, 0));
            ImGui::TextWrapped("%s", status.c_str());
            if (provider != 0 && total_mb)
                ImGui::Text("Device memory: %zu / %zu MiB", used_mb, total_mb);
            if (selected >= 0)
            {
                const auto& t = results[selected].image.timings;
                ImGui::Separator();
                ImGui::Text("Encode %.0f ms | RNG %.0f ms", t.encode_ms, t.rng_ms);
                ImGui::Text("Denoise %.0f ms (%.0f ms/step)", t.denoise_ms, t.per_step_ms);
                ImGui::Text("Decode %.0f ms | Total %.2f s", t.decode_ms, t.total_ms / 1000);
            }
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginChild("images", ImVec2(0, 0), true);
            for (auto& result : results)
                if (!result.texture)
                    Upload(result);
            ImGui::BeginChild("preview", ImVec2(0, -115), false);
            if (selected >= 0)
            {
                const auto available = ImGui::GetContentRegionAvail();
                const float side = std::max(1.0f, std::min(available.x, available.y));
                ImGui::Image(static_cast<ImTextureID>(results[selected].texture), ImVec2(side, side));
            }
            else
                ImGui::TextDisabled("Generated images will appear here.");
            ImGui::EndChild();
            ImGui::BeginChild("thumbnails", ImVec2(0, 110), false, ImGuiWindowFlags_HorizontalScrollbar);
            for (int i = 0; i < static_cast<int>(results.size()); ++i)
            {
                if (i)
                    ImGui::SameLine();
                ImGui::PushID(i);
                ImGui::BeginGroup();
                if (ImGui::ImageButton("image", static_cast<ImTextureID>(results[i].texture), ImVec2(72, 72)))
                    selected = i;
                ImGui::Text("%u", results[i].seed);
                ImGui::EndGroup();
                ImGui::PopID();
            }
            ImGui::EndChild();
            ImGui::EndChild();
            ImGui::End();
            ImGui::Render();
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            glViewport(0, 0, width, height);
            glClearColor(0.09f, 0.09f, 0.10f, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
            if (smoke && ++frames >= 3)
                break;
        }
        for (auto& result : results)
            if (result.texture)
                glDeleteTextures(1, &result.texture);
        return exit_code;
    }
} // namespace

int main(int argc, char** argv)
{
    GLFWwindow* window = nullptr;
    bool imgui = false, glfw_backend = false, gl_backend = false;
    int result = 1;
    try
    {
        argparse::ArgumentParser args("din_flux2_ui");
        args.add_argument("--model-dir").default_value(DEFAULT_MODEL_BASE_PATH.string());
        args.add_argument("--output").default_value(std::string{});
        args.add_argument("--ep-cache").default_value(Flux2Config{}.ep_cache_dir.string());
        args.add_argument("--ep-context-dir").default_value(Flux2Config{}.ep_context_dir.string());
        args.add_argument("--prompt").default_value(std::string(DEFAULT_PROMPT));
        args.add_argument("--encoder").default_value(std::string("auto")).choices("auto", "4b", "translator");
        args.add_argument("--precision").default_value(std::string("bf16")).choices("bf16", "fp16", "fp8", "nvfp4");
        args.add_argument("--smoke-test").default_value(false).implicit_value(true);
        args.add_argument("--auto-generate").default_value(false).implicit_value(true);
        args.add_argument("--exit-after-generation").default_value(false).implicit_value(true);
        args.parse_args(argc, argv);
        Flux2Config config;
        config.model_dir = args.get<std::string>("--model-dir");
        config.output_path = args.get<std::string>("--output");
        config.ep_cache_dir = args.get<std::string>("--ep-cache");
        config.ep_context_dir = args.get<std::string>("--ep-context-dir");
        config.prompt = args.get<std::string>("--prompt");
        config.precision = args.get<std::string>("--precision");
        config.num_images = 1;
        const auto encoder = args.get<std::string>("--encoder");
        const bool translator = encoder == "translator" ||
        (encoder == "auto" && std::filesystem::is_regular_file(
            config.model_dir / "text_encoder_translator/model.onnx"));
        config.text_encoder = translator ? Flux2TextEncoder::Qwen3_06BTranslator : Flux2TextEncoder::Qwen3_4B;
        config.weight_streaming_budget = "-1";
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        window = glfwCreateWindow(1280, 850, "DIN Flux2 Studio", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("OpenGL 3.2 window creation failed");
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        imgui = true;
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        glfw_backend = ImGui_ImplGlfw_InitForOpenGL(window, true);
        gl_backend = ImGui_ImplOpenGL3_Init("#version 130");
        if (!glfw_backend || !gl_backend)
            throw std::runtime_error("ImGui graphics initialization failed");
        result = Studio(config, args.get<bool>("--smoke-test"), args.get<bool>("--auto-generate"),
                        args.get<bool>("--exit-after-generation"));
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
    }
    if (gl_backend)
        ImGui_ImplOpenGL3_Shutdown();
    if (glfw_backend)
        ImGui_ImplGlfw_Shutdown();
    if (imgui)
        ImGui::DestroyContext();
    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
