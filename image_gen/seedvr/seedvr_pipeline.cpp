// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "seedvr_pipeline.h"
#ifdef DIN_SEEDVR_NATIVE_VAE
#include "seedvr_vae.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "nvtx_helper.h"
#include "ort_session.h"
#include "rng.h"
#include "rng_cuda.h"
#include "seedvr_cuda_kernels.h"
#include "seedvr_metadata.h"
#include "seedvr_tiling.h"
#include <onnxruntime_run_options_config_keys.h>

namespace din::seedvr
{
namespace detail
{
inline constexpr size_t DivRoundUp(size_t value, size_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

using NativeVaeRun = std::function<void(const void*, void*, int, int)>;
// Fixed tile geometry, bindings and scratch, reused for every call.
class VaeExecution
{
public:
    VaeExecution(common::OrtRunner& runner, Ort::Value& input, Ort::Value& output, size_t height, size_t width,
                 size_t tile_height, size_t tile_width, size_t overlap, bool encode, size_t frames, NativeVaeRun native)
        : runner_(runner)
        , input_(input)
        , output_(output)
        , height_(height)
        , width_(width)
        , th_(tile_height)
        , tw_(tile_width)
        , native_(std::move(native))
        , binding_(runner.session)
    {
        const size_t lt = (frames - 1) / 4 + 1;
        const size_t ti = encode ? frames : lt, to = encode ? lt : frames;
        ci_ = (encode ? 3 : kLatentChannels) * ti;
        co_ = (encode ? kLatentChannels : 3) * to;
        numerator_ = encode ? 1 : 8;
        denominator_ = encode ? 8 : 1;
        oh_ = height * numerator_ / denominator_;
        ow_ = width * numerator_ / denominator_;
        tiles_ = MakeVaeTiles(height, width, tile_height, tile_width, overlap);
        if (tiles_.size() > 1)
        {
            std::vector<int64_t> si{1, static_cast<int64_t>(ci_), static_cast<int64_t>(th_), static_cast<int64_t>(tw_)};
            std::vector<int64_t> so{1, static_cast<int64_t>(co_), static_cast<int64_t>(th_ * numerator_ / denominator_),
                                    static_cast<int64_t>(tw_ * numerator_ / denominator_)};
            if (frames > 1)
            {
                si[1] /= ti;
                si.insert(si.begin() + 2, ti);
                so[1] /= to;
                so.insert(so.begin() + 2, to);
            }
            auto allocator = runner.DeviceAllocator();
            tile_in_ = Ort::Value::CreateTensor<Fp16>(allocator, si.data(), si.size());
            tile_out_ = Ort::Value::CreateTensor<Fp16>(allocator, so.data(), so.size());
            const int64_t ns = co_ * oh_ * ow_, nw = oh_ * ow_;
            sums_ = Ort::Value::CreateTensor<float>(allocator, &ns, 1);
            weights_ = Ort::Value::CreateTensor<float>(allocator, &nw, 1);
        }
        if (!native_)
        {
            binding_.BindInput(encode ? "image" : "latent", tiles_.size() > 1 ? tile_in_ : input_);
            binding_.BindOutput(encode ? "latent" : "image", tiles_.size() > 1 ? tile_out_ : output_);
        }
    }
    void Run(Ort::RunOptions& options, cudaStream_t stream)
    {
        if (tiles_.size() == 1)
        {
            if (native_)
                native_(input_.GetTensorData<Fp16>(), output_.GetTensorMutableData<Fp16>(), height_, width_);
            else
                runner_.session.Run(options, binding_);
            return;
        }
        float* sum = sums_.GetTensorMutableData<float>();
        float* weight = weights_.GetTensorMutableData<float>();
        SEEDVR_CUDA_CHECK(cudaMemsetAsync(sum, 0, co_ * oh_ * ow_ * sizeof(float), stream));
        SEEDVR_CUDA_CHECK(cudaMemsetAsync(weight, 0, oh_ * ow_ * sizeof(float), stream));
        for (const auto& tile : tiles_)
        {
            LaunchSeedVrGatherTile(stream, input_.GetTensorData<Fp16>(), tile_in_.GetTensorMutableData<Fp16>(), ci_,
                                   height_, width_, tile);
            if (native_)
                native_(tile_in_.GetTensorData<Fp16>(), tile_out_.GetTensorMutableData<Fp16>(), th_, tw_);
            else
                runner_.session.Run(options, binding_);
            LaunchSeedVrBlendTile(stream, tile_out_.GetTensorData<Fp16>(), sum, weight, co_, oh_, ow_,
                                  ScaleVaeTile(tile, numerator_, denominator_));
        }
        LaunchSeedVrNormalizeTiles(stream, sum, weight, output_.GetTensorMutableData<Fp16>(), co_, oh_ * ow_);
    }

private:
    common::OrtRunner& runner_;
    Ort::Value &input_, &output_;
    size_t height_, width_, th_, tw_, ci_, co_, oh_, ow_, numerator_, denominator_;
    NativeVaeRun native_;
    std::vector<SeedVrTile> tiles_;
    Ort::Value tile_in_{nullptr}, tile_out_{nullptr}, sums_{nullptr}, weights_{nullptr};
    Ort::IoBinding binding_;
};

struct MetadataBuffers
{
    common::TensorBuffer<int64_t> wt, ws, ct, cs, pf, pg;
    common::TensorBuffer<bool> valid;
    std::unique_ptr<common::TensorBuffer<Fp16>> freqs;
    std::unique_ptr<common::TensorBuffer<float>> freqs_fp32;
    MetadataBuffers(common::OrtRunner& r, Ort::IoBinding& io, const char* name, const MetadataDimensions& dimensions,
                    int64_t latent_height, int64_t latent_width, bool shifted, int64_t temporal, bool version_7b,
                    bool fp32_rope)
        : wt(r, {dimensions.video_tokens}, true)
        , ws(r, {dimensions.video_tokens}, true)
        , ct(r, {dimensions.attention_tokens}, true)
        , cs(r, {dimensions.attention_tokens}, true)
        , pf(r, {dimensions.attention_tokens}, true)
        , pg(r, {dimensions.padded_tokens}, true)
        , valid(r, {dimensions.window_count, dimensions.max_window_length}, true)
    {
        try
        {
            const std::vector<int64_t> shape{dimensions.attention_tokens, version_7b ? 30 : kRopePairs, 2, 2};
            if (fp32_rope)
            {
                freqs_fp32 = std::make_unique<common::TensorBuffer<float>>(r, shape, true);
                MakeMetadata(latent_height, latent_width, shifted, temporal, version_7b, io, name,
                             wt, ws, ct, cs, pf, pg, valid, *freqs_fp32);
            }
            else
            {
                freqs = std::make_unique<common::TensorBuffer<Fp16>>(r, shape, true);
                MakeMetadata(latent_height, latent_width, shifted, temporal, version_7b, io, name,
                             wt, ws, ct, cs, pf, pg, valid, *freqs);
            }
        }
        catch (...)
        {
            // Complete partial uploads before constructor unwinding releases their buffers.
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(r.compute_stream->GetHandle()));
            throw;
        }
    }
};

inline void ValidateOptions(const DynamicOptions& o)
{
    if (!std::isfinite(o.latent_noise_scale) || o.latent_noise_scale < 0 || o.latent_noise_scale > 1)
        throw std::invalid_argument("SeedVR latent noise scale must be finite and in [0, 1]");
    if (o.color_correction != "lab" && o.color_correction != "none")
        throw std::invalid_argument("SeedVR color correction must be lab or none");
    if (o.rng != common::Rng::CpuStd && o.rng != common::Rng::CpuPhilox && o.rng != common::Rng::CudaPhilox)
        throw std::invalid_argument("Invalid SeedVR RNG");
}

class PipelineCore
{
public:
    PipelineCore(ModelConfig config, ImageShape shape, size_t frame_count, DynamicOptions options)
        : config_(std::move(config))
        , options_(std::move(options))
        , shape_(shape)
        , frames(frame_count)
    {
        const auto start = std::chrono::steady_clock::now();
        ValidateOptions(options_);
        if (!frames || (frames - 1) % 4 || frames > static_cast<size_t>(INT_MAX))
            throw std::invalid_argument("SeedVR frame count must be 4n+1 and fit an int");
        if (!shape.source_height || !shape.source_width || shape.source_height > INT_MAX ||
            shape.source_width > INT_MAX ||
            shape.source_height > std::numeric_limits<size_t>::max() / (3 * sizeof(float)) / shape.source_width ||
            shape.target_height < 2 || shape.target_width < 2 || shape.target_height > 3840 ||
            shape.target_width > 3840)
            throw std::invalid_argument("Invalid SeedVR source or target dimensions");
        video = frames > 1;
        native_vae = config_.vae_backend == "cudnn";
        if (config_.vae_backend != "onnx" && !native_vae)
            throw std::invalid_argument("VAE backend must be onnx or cudnn");
        if (native_vae && config_.vae_weights_dir.empty())
            throw std::invalid_argument("Native VAE requires weights");
#ifndef DIN_SEEDVR_NATIVE_VAE
        if (native_vae)
            throw std::invalid_argument("Rebuild with DIN_SEEDVR_NATIVE_VAE=ON");
#endif
        if (config_.vae_workspace_mib < 0 || config_.vae_tuning_candidates < 1)
            throw std::invalid_argument("Invalid SeedVR VAE workspace/tuning options");
        if (config_.transformer_precision != "fp16" && config_.transformer_precision != "bf16" && config_.transformer_precision != "fp8_wo" &&
            config_.transformer_precision != "nvfp4")
            throw std::invalid_argument("Invalid transformer precision");
        if (video && config_.transformer_precision != "fp16" && config_.transformer_precision != "bf16")
            throw std::invalid_argument("Video DiT requires BF16 or FP16");
        ValidateVaeTiling(config_.vae_encoder_tile_size, config_.vae_encoder_overlap);
        ValidateVaeTiling(config_.vae_decoder_tile_size, config_.vae_decoder_overlap);
        padded_height = DivRoundUp(shape.target_height, 16) * 16;
        padded_width = DivRoundUp(shape.target_width, 16) * 16;
        const bool portrait = padded_height > padded_width;
        if (padded_height < 32 || padded_width < 32 || padded_height > (portrait ? 3840 : 2160) ||
            padded_width > (portrait ? 2160 : 3840))
            throw std::invalid_argument("SeedVR padded shape must fit the 32..3840x2160 profile");
        latent_frames = (frames - 1) / 4 + 1;
        latent_height = padded_height / 8;
        latent_width = padded_width / 8;
        encoder_height = config_.vae_encoder_tile_size ? std::min(padded_height, size_t(config_.vae_encoder_tile_size))
                                                       : padded_height;
        encoder_width = config_.vae_encoder_tile_size ? std::min(padded_width, size_t(config_.vae_encoder_tile_size))
                                                      : padded_width;
        decoder_height = config_.vae_decoder_tile_size
                             ? std::min(padded_height, size_t(config_.vae_decoder_tile_size)) / 8
                             : latent_height;
        decoder_width = config_.vae_decoder_tile_size
                            ? std::min(padded_width, size_t(config_.vae_decoder_tile_size)) / 8
                            : latent_width;
        spatial_size = latent_frames * latent_height * latent_width;
        noise_count = kLatentChannels * spatial_size;
        const bool version_7b = !video && fs::is_regular_file(
            config_.model_dir / ("seedvr2_7b_" + config_.transformer_precision + "_dit.onnx"));
        if (version_7b && config_.transformer_precision != "fp16" && config_.transformer_precision != "bf16")
            throw std::invalid_argument("SeedVR2 7B currently supports FP16 or BF16.");
        const std::string variant = version_7b ? "7b" : "3b";
        const auto normal_profile = ProfileMetadata(padded_height, padded_width, false, latent_frames);
        const auto shifted_profile = ProfileMetadata(padded_height, padded_width, true, latent_frames);
        const bool fp32_rope = version_7b || config_.transformer_precision == "bf16";
        const fs::path context_path = config_.model_dir / ("seedvr2_" + variant + "_fp16_positive_context.bin");
        const std::vector<Fp16> context_values = LoadContext(context_path);
        const fs::path encoder_path =
            config_.model_dir / (video ? "seedvr2_video_vae_encoder.onnx" : "seedvr2_ema_vae_encoder.onnx");
        const fs::path dit_path =
            config_.model_dir /
            (video ? "seedvr2_video_dit.onnx" : "seedvr2_" + variant + "_" + config_.transformer_precision + "_dit.onnx");
        const fs::path decoder_path =
            config_.model_dir / (video ? "seedvr2_video_vae_decoder.onnx" : "seedvr2_ema_vae_decoder.onnx");
        const std::string encoder_label =
            "SeedVR2 / VAE encoder / " +
            (native_vae ? "cuDNN / " + config_.vae_weights_dir.generic_string() : encoder_path.generic_string());
        const std::string dit_label = "SeedVR2 / DiT / " + dit_path.generic_string();
        const std::string decoder_label =
            "SeedVR2 / VAE decoder / " +
            (native_vae ? "cuDNN / " + config_.vae_weights_dir.generic_string() : decoder_path.generic_string());
        encoder_label_ = encoder_label;
        dit_label_ = dit_label;
        decoder_label_ = decoder_label;
        clip_label_ = "SeedVR2 / Clip / " + std::to_string(frames) + " frames / " + config_.model_dir.generic_string();

        if (!sessions_)
        {
            [[maybe_unused]] common::nvtx_scoped_range load_range{"SeedVR2 / Load models"};
            const auto session_start = std::chrono::steady_clock::now();
            std::cout << "Transformer precision: " << config_.transformer_precision << " (" << dit_path.generic_string()
                      << ")" << std::endl;
            if (!native_vae)
                RequireFile(encoder_path);
            RequireFile(dit_path);
            if (!native_vae)
                RequireFile(decoder_path);
            std::cout << "Creating SeedVR sessions for padded " << padded_width << 'x' << padded_height << std::endl;
            const auto trt_device = common::RegisterTensorRTRTXProvider(env_);
            cuda_device = common::ChooseCudaDeviceOrdinal(trt_device, nullptr);
            SEEDVR_CUDA_CHECK(cudaSetDevice(cuda_device));
            cudaDeviceProp properties{};
            SEEDVR_CUDA_CHECK(cudaGetDeviceProperties(&properties, cuda_device));
            std::cout << "SeedVR CUDA device " << cuda_device << ": " << properties.name << std::endl;
            compute_stream_ = common::CreateTensorRTRTXComputeStream(env_);
            auto* stream = &compute_stream_;
            const common::EpContextOptions ep_context{.output_dir = config_.ep_context_dir.string()};
            const std::string video_prefix = video ? "video" + std::to_string(frames) + "_" : "";
            const std::string image_size =
                video_prefix + std::to_string(padded_width) + "x" + std::to_string(padded_height);

            auto sessions = std::make_unique<SessionBundle>();
            sessions->padded_height = padded_height;
            sessions->padded_width = padded_width;
            // TODO: TensorRT RTX attention decomposition fails with dynamic spatial shapes;
            // restore ranged profiles when that path supports them.
            auto encoder_profile = ExactProfile(
                "image:" + (video ? ShapeString({1, 3, static_cast<int64_t>(frames), encoder_height, encoder_width})
                                  : ShapeString({1, 3, encoder_height, encoder_width})),
                video_prefix + "vae_encoder_" + std::to_string(encoder_width) + "x" + std::to_string(encoder_height));
            encoder_profile.skip_compile = true;
            if (!native_vae)
            {
                [[maybe_unused]] common::nvtx_scoped_range model_range{encoder_label.c_str()};
                sessions->encoder = std::make_unique<common::OrtRunner>(env_, encoder_path.string(), "trt-rtx",
                                                                        config_.ep_cache_dir.string(), ep_context,
                                                                        encoder_profile, stream);
            }
            auto dit_profile =
                ExactProfile(DitProfile(static_cast<int64_t>(padded_height), static_cast<int64_t>(padded_width),
                                        normal_profile, shifted_profile, latent_frames, version_7b ? 30 : kRopePairs),
                             "dit_" + config_.transformer_precision + "_" + image_size);
            // The exporter gives each dynamic ONNX input dimension a stable name.
            // Specializing them at session creation avoids a TensorRT RTX Myelin
            // shape-inference failure when the DiT uses the full resolution range.
            dit_profile.free_dimension_overrides = {
                {"latent_dim3", latent_height},
                {"latent_dim4", latent_width},
                {"normal_window_tgt_idx_dim0", normal_profile.video_tokens},
                {"normal_window_src_idx_dim0", normal_profile.video_tokens},
                {"normal_concat_tgt_idx_dim0", normal_profile.attention_tokens},
                {"normal_concat_src_idx_dim0", normal_profile.attention_tokens},
                {"normal_padded_flat_idx_dim0", normal_profile.attention_tokens},
                {"normal_padded_gather_idx_dim0", normal_profile.padded_tokens},
                {"normal_valid_keys_dim0", normal_profile.window_count},
                {"normal_valid_keys_dim1", normal_profile.max_window_length},
                {"shifted_window_tgt_idx_dim0", shifted_profile.video_tokens},
                {"shifted_concat_tgt_idx_dim0", shifted_profile.attention_tokens},
                {"shifted_concat_src_idx_dim0", shifted_profile.attention_tokens},
                {"shifted_padded_flat_idx_dim0", shifted_profile.attention_tokens},
                {"shifted_padded_gather_idx_dim0", shifted_profile.padded_tokens},
                {"shifted_valid_keys_dim0", shifted_profile.window_count},
                {"shifted_valid_keys_dim1", shifted_profile.max_window_length}};
            if (!config_.weight_streaming_budget.empty())
            {
                dit_profile.extra_ep_options.emplace_back("nv_weight_streaming_budget",
                                                          config_.weight_streaming_budget);
            }
            std::cerr << "[SeedVR initialization] creating DiT ORT session" << std::endl;
            {
                [[maybe_unused]] common::nvtx_scoped_range model_range{dit_label.c_str()};
                sessions->dit = std::make_unique<common::OrtRunner>(
                    env_, dit_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context, dit_profile, stream);
            }
            std::cerr << "[SeedVR initialization] DiT ORT session ready" << std::endl;
            if (!native_vae)
            {
                [[maybe_unused]] common::nvtx_scoped_range model_range{decoder_label.c_str()};
                sessions->decoder = std::make_unique<common::OrtRunner>(
                    env_, decoder_path.string(), "trt-rtx", config_.ep_cache_dir.string(), ep_context,
                    ExactProfile("latent:" + (video ? ShapeString({1, kLatentChannels, latent_frames, decoder_height,
                                                                   decoder_width})
                                                    : ShapeString({1, kLatentChannels, decoder_height, decoder_width})),
                                 video_prefix + std::string("vae_decoder_unclamped_") +
                                     std::to_string(decoder_width * 8) + "x" + std::to_string(decoder_height * 8)),
                    stream);
            }
            if (!sessions->dit->HasDeviceIo() ||
                (!native_vae && (!sessions->encoder->HasDeviceIo() || !sessions->decoder->HasDeviceIo())))
            {
                throw std::runtime_error("SeedVR requires CUDA-backed TensorRT RTX I/O");
            }
#ifdef DIN_SEEDVR_NATIVE_VAE
            if (native_vae)
            {
                [[maybe_unused]] common::nvtx_scoped_range model_range{"SeedVR2 / Load VAE / cuDNN"};
                NativeVae::Shapes shapes;
                shapes.encoder = NativeVae::Shape{static_cast<int>(frames), static_cast<int>(encoder_height),
                                                  static_cast<int>(encoder_width)};
                shapes.decoder = NativeVae::Shape{static_cast<int>(latent_frames), static_cast<int>(decoder_height),
                                                  static_cast<int>(decoder_width)};
                sessions->native = std::make_unique<NativeVae>(
                    config_.vae_weights_dir, reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle()), shapes,
                    size_t(config_.vae_workspace_mib) << 20, config_.vae_autotune, config_.vae_tuning_cache,
                    config_.vae_tuning_candidates);
                const auto memory = sessions->native->GetWorkingMemoryStats();
                std::cout << "VAE encoder physical working memory required: " << memory.encoder_bytes << " bytes\n"
                          << "VAE decoder physical working memory required: " << memory.decoder_bytes << " bytes\n"
                          << "VAE shared physical working memory capacity (max, not sum): " << memory.shared_bytes
                          << " bytes\n"
                          << "VAE resident physical working memory: " << memory.resident_bytes << " bytes\n"
                          << "VAE encoder persistent I/O (outside VMM): " << memory.encoder_io_bytes << " bytes\n"
                          << "VAE decoder persistent I/O (outside VMM): " << memory.decoder_io_bytes << " bytes"
                          << std::endl;
            }
#endif
            sessions_ = std::move(sessions);
            const double session_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - session_start).count();
            std::cout << "SeedVR session creation: " << session_ms << " ms" << std::endl;
        }

        try
        {
            auto& encoder = *(native_vae ? sessions_->dit : sessions_->encoder);
            auto& dit = *sessions_->dit;
            auto& decoder = *(native_vae ? sessions_->dit : sessions_->decoder);
            std::vector<int64_t> pixel_shape{1, 3, static_cast<int64_t>(padded_height), static_cast<int64_t>(padded_width)};
            std::vector<int64_t> encoded_shape{1, kLatentChannels, latent_height, latent_width};
            const std::vector<int64_t> latent_shape{1, kLatentChannels, latent_frames, latent_height, latent_width};
            const std::vector<int64_t> condition_shape{1, kLatentChannels + 1, latent_frames, latent_height, latent_width};
            if (video)
            {
                pixel_shape.insert(pixel_shape.begin() + 2, frames);
                encoded_shape.insert(encoded_shape.begin() + 2, latent_frames);
            }
            auto make_half = [](common::OrtRunner& r, const std::vector<int64_t>& s)
            {
                return std::make_unique<common::TensorBuffer<Fp16>>(r, s, true);
            };
            pixels_ = make_half(encoder, pixel_shape);
            encoded_ = make_half(encoder, encoded_shape);  // Reused for sampled latent.
            condition_ = make_half(dit, condition_shape);
            latent_ = make_half(dit, latent_shape);
            velocity_ = make_half(dit, latent_shape);
            context_ = make_half(dit, {1, kContextTokens, kContextWidth});
            timestep_ = make_half(dit, {1});
            auto image_allocator = encoder.DeviceAllocator();
            const std::array<int64_t, 3> input_shape{3, static_cast<int64_t>(shape.source_height),
                                                   static_cast<int64_t>(shape.source_width)};
            input_device_ = Ort::Value::CreateTensor<float>(image_allocator, input_shape.data(), input_shape.size());
            const std::array<int64_t, 4> reference_shape{static_cast<int64_t>(frames), 3,
                                                       static_cast<int64_t>(padded_height),
                                                       static_cast<int64_t>(padded_width)};
            reference_ = Ort::Value::CreateTensor<float>(image_allocator, reference_shape.data(), reference_shape.size());
            const int64_t noise_elements = static_cast<int64_t>(noise_count);
            noise_ = Ort::Value::CreateTensor<float>(image_allocator, &noise_elements, 1);
            auto pinned_allocator = dit.PinnedAllocator();
            const std::array<int64_t, 2> noise_host_shape{2, noise_elements};
            noise_host_ = Ort::Value::CreateTensor<float>(pinned_allocator, noise_host_shape.data(), noise_host_shape.size());
            const std::array<int64_t, 4> output_shape{static_cast<int64_t>(frames), 3,
                                                    static_cast<int64_t>(shape.target_height / 2 * 2),
                                                    static_cast<int64_t>(shape.target_width / 2 * 2)};
            output_device_ = Ort::Value::CreateTensor<float>(image_allocator, output_shape.data(), output_shape.size());
            output_host_ = Ort::Value::CreateTensor<float>(pinned_allocator, output_shape.data(), output_shape.size());
            dit_io_ = std::make_unique<Ort::IoBinding>(dit.session);
            dit_io_->BindInput("latent", latent_->BindingValue());
            dit_io_->BindInput("condition", condition_->BindingValue());
            dit_io_->BindInput("timestep", timestep_->BindingValue());
            dit_io_->BindInput("context", context_->BindingValue());
            dit_io_->BindOutput("denoised_latent", velocity_->BindingValue());
            normal_ = std::make_unique<MetadataBuffers>(dit, *dit_io_, "normal", normal_profile, latent_height,
                                                        latent_width, false, latent_frames, version_7b, fp32_rope);
            shifted_ = std::make_unique<MetadataBuffers>(dit, *dit_io_, "shifted", shifted_profile, latent_height,
                                                         latent_width, true, latent_frames, version_7b, fp32_rope);
            std::copy(context_values.begin(), context_values.end(), context_->HostData());
            context_->CopyAsyncToDevice();
            timestep_->HostData()[0] = Fp16(1000.0f);
            timestep_->CopyAsyncToDevice();
            run_options_.SetSyncStream(compute_stream_);
            run_options_.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
            NativeVaeRun encode, decode;
    #ifdef DIN_SEEDVR_NATIVE_VAE
            if (native_vae)
            {
                encode = [this](const void* x, void* y, int, int)
                {
                    sessions_->native->Encode(x, y);
                };
                decode = [this](const void* x, void* y, int, int)
                {
                    sessions_->native->Decode(x, y);
                };
            }
    #endif
            encoder_exec_ = std::make_unique<VaeExecution>(encoder, pixels_->BindingValue(), encoded_->BindingValue(),
                                                           padded_height, padded_width, encoder_height, encoder_width,
                                                           config_.vae_encoder_tile_size ? config_.vae_encoder_overlap : 0,
                                                           true, frames, std::move(encode));
            decoder_exec_ = std::make_unique<VaeExecution>(
                decoder, encoded_->BindingValue(), pixels_->BindingValue(), latent_height, latent_width, decoder_height,
                decoder_width, config_.vae_decoder_tile_size ? config_.vae_decoder_overlap / 8 : 0, false, frames,
                std::move(decode));
            lab_scratch_bytes_ = SeedVrLabScratchBytes(shape.target_height, shape.target_width);
            const int64_t scratch_bytes = lab_scratch_bytes_;
            auto allocator = dit.DeviceAllocator();
            lab_scratch_ = Ort::Value::CreateTensor<uint8_t>(allocator, &scratch_bytes, 1);
            SEEDVR_CUDA_CHECK(cudaStreamSynchronize(Stream()));
            std::cout << "SeedVR pipeline construction: "
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                      << " ms" << std::endl;
        }
        catch (...)
        {
            cudaStreamSynchronize(Stream());
            throw;
        }
    }

    ~PipelineCore()
    {
        if (compute_stream_)
        {
            cudaStreamSynchronize(Stream());
        }
    }

    Ort::Value CreateImage(ImageRole role, ImageDevice device)
    {
        const auto s = PublicShape(role);
        if (device == ImageDevice::Cpu)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            return Ort::Value::CreateTensor<float>(allocator, s.data(), s.size());
        }
        if (device != ImageDevice::Cuda)
            throw std::invalid_argument("Invalid image device");
        auto allocator = sessions_->dit->DeviceAllocator();
        return Ort::Value::CreateTensor<float>(allocator, s.data(), s.size());
    }

    void UpdateOptions(const DynamicOptionsUpdate& update)
    {
        auto next = options_;
        if (update.seed)
            next.seed = *update.seed;
        if (update.rng)
            next.rng = *update.rng;
        if (update.latent_noise_scale)
            next.latent_noise_scale = *update.latent_noise_scale;
        if (update.color_correction)
            next.color_correction = *update.color_correction;
        ValidateOptions(next);
        if (update.weight_streaming_budget)
        {
            if (config_.weight_streaming_budget.empty() || config_.weight_streaming_budget == "0")
                throw std::invalid_argument("Weight streaming must be enabled during construction");
            if (update.weight_streaming_budget->empty())
                throw std::invalid_argument(
                    "Weight-streaming budget must not be empty; use 0 to disable resident streaming");
            SEEDVR_CUDA_CHECK(cudaStreamSynchronize(Stream()));
            const char* keys[]{"nv_weight_streaming_budget"};
            const char* values[]{update.weight_streaming_budget->c_str()};
            // Provider validates the budget and rejects unsupported dynamic updates.
            sessions_->dit->session.SetEpDynamicOptions(keys, values, 1);
        }
        options_ = std::move(next);
    }

    void Run(std::span<const Ort::Value> inputs, std::span<Ort::Value> outputs)
    {
        [[maybe_unused]] common::nvtx_scoped_range clip_range{clip_label_.c_str()};
        if (inputs.size() != frames || outputs.size() != frames)
            throw std::invalid_argument("SeedVR requires exactly the configured number of input/output frames");
        for (size_t t = 0; t < frames; ++t)
        {
            ValidateTensor(inputs[t], ImageRole::Input);
            ValidateTensor(outputs[t], ImageRole::Output);
        }
        const auto start = std::chrono::steady_clock::now();
        const size_t plane = padded_height * padded_width;
        const size_t rh = shape_.target_height / 2 * 2, rw = shape_.target_width / 2 * 2;
        const size_t output_elements = 3 * rh * rw;
        try
        {
            for (size_t t = 0; t < frames; ++t)
            {
                const float* input = inputs[t].GetTensorData<float>();
                if (IsCpu(inputs[t]))
                {
                    auto* uploaded = input_device_.GetTensorMutableData<float>();
                    SEEDVR_CUDA_CHECK(cudaMemcpyAsync(uploaded, input,
                                                      3 * shape_.source_height * shape_.source_width * sizeof(float),
                                                      cudaMemcpyHostToDevice, Stream()));
                    input = uploaded;
                }
                LaunchSeedVrResizeBicubic(Stream(), input, pixels_->BindingValue().GetTensorMutableData<Fp16>() + t * plane,
                                          reference_.GetTensorMutableData<float>() + 3 * t * plane,
                                          shape_.source_height, shape_.source_width, shape_.target_height,
                                          shape_.target_width, padded_height, padded_width, frames * plane);
            }
            {
                [[maybe_unused]] common::nvtx_scoped_range range{encoder_label_.c_str()};
                encoder_exec_->Run(run_options_, Stream());
            }
            LaunchSeedVrConditionKernel(Stream(), encoded_->BindingValue().GetTensorData<Fp16>(),
                                        condition_->BindingValue().GetTensorMutableData<Fp16>(), spatial_size);
            if (options_.rng == common::Rng::CudaPhilox)
            {
                common::GenerateNormalCuda(Stream(), noise_.GetTensorMutableData<float>(), noise_count,
                                           options_.seed);
            }
            else
            {
                auto* host_noise = noise_host_.GetTensorMutableData<float>();
                common::GenerateNormalCpu(std::span<float>(host_noise, noise_count), options_.seed, options_.rng);
                SEEDVR_CUDA_CHECK(cudaMemcpyAsync(noise_.GetTensorMutableData<float>(), host_noise,
                                                  noise_count * sizeof(float), cudaMemcpyHostToDevice, Stream()));
            }
            LaunchSeedVrFloatToHalfKernel(Stream(), noise_.GetTensorData<float>(),
                                          latent_->BindingValue().GetTensorMutableData<Fp16>(), noise_count);
            if (options_.latent_noise_scale > 0)
            {
                const uint64_t seed = uint64_t(options_.seed) + 1;
                if (options_.rng == common::Rng::CudaPhilox)
                    common::GenerateNormalCuda(Stream(), noise_.GetTensorMutableData<float>(), noise_count,
                                               seed);
                else
                {
                    // Keep base and augmentation host data separate until their uploads complete.
                    auto* host_noise = noise_host_.GetTensorMutableData<float>() + noise_count;
                    common::GenerateNormalCpu(std::span<float>(host_noise, noise_count), seed, options_.rng);
                    SEEDVR_CUDA_CHECK(cudaMemcpyAsync(noise_.GetTensorMutableData<float>(), host_noise,
                                                      noise_count * sizeof(float), cudaMemcpyHostToDevice, Stream()));
                }
                const float area = static_cast<float>(padded_height * padded_width);
                const float shift = video
                                        ? 1.0f + (area * static_cast<float>(frames) - 256.0f * 256.0f * 37.0f) *
                                                     (4.0f / (1280.0f * 720.0f * 145.0f - 256.0f * 256.0f * 37.0f))
                                        : 1.0f + (area - 256.0f * 256.0f) * (2.2f / (1024.0f * 1024.0f - 256.0f * 256.0f));
                const float amount = options_.latent_noise_scale;
                const float sigma = shift * amount / (1.0f + (shift - 1.0f) * amount);
                LaunchSeedVrConditionNoiseKernel(Stream(), condition_->BindingValue().GetTensorMutableData<Fp16>(),
                                                 latent_->BindingValue().GetTensorData<Fp16>(),
                                                 noise_.GetTensorData<float>(), sigma, noise_count);
            }
            {
                [[maybe_unused]] common::nvtx_scoped_range range{dit_label_.c_str()};
                sessions_->dit->session.Run(run_options_, *dit_io_);
            }
            LaunchSeedVrEulerKernel(Stream(), latent_->BindingValue().GetTensorData<Fp16>(),
                                    velocity_->BindingValue().GetTensorData<Fp16>(),
                                    encoded_->BindingValue().GetTensorMutableData<Fp16>(), noise_count);
            {
                [[maybe_unused]] common::nvtx_scoped_range range{decoder_label_.c_str()};
                decoder_exec_->Run(run_options_, Stream());
            }
            for (size_t t = 0; t < frames; ++t)
            {
                auto* image = pixels_->BindingValue().GetTensorMutableData<Fp16>() + t * plane;
                if (options_.color_correction == "lab")
                {
                    LaunchSeedVrLabKernel(Stream(), image,
                                          reference_.GetTensorData<float>() + 3 * t * plane, shape_.target_height,
                                          shape_.target_width, padded_width, frames * plane, plane,
                                          lab_scratch_.GetTensorMutableData<uint8_t>(),
                                          lab_scratch_bytes_);
                }
                float* dst = IsCpu(outputs[t]) ? output_device_.GetTensorMutableData<float>() + t * output_elements
                                              : outputs[t].GetTensorMutableData<float>();
                LaunchSeedVrCropToFloat(Stream(), image, dst, rh, rw, padded_width, frames * plane);
                if (IsCpu(outputs[t]))
                {
                    SEEDVR_CUDA_CHECK(cudaMemcpyAsync(output_host_.GetTensorMutableData<float>() + t * output_elements,
                                                      dst, output_elements * sizeof(float), cudaMemcpyDeviceToHost, Stream()));
                }
            }
            SEEDVR_CUDA_CHECK(cudaStreamSynchronize(Stream()));
        }
        catch (...)
        {
            // Finish queued work before the caller can release or reuse its tensors.
            cudaStreamSynchronize(Stream());
            throw;
        }
        for (size_t t = 0; t < frames; ++t)
            if (IsCpu(outputs[t]))
                std::memcpy(outputs[t].GetTensorMutableData<float>(), output_host_.GetTensorData<float>() + t * output_elements,
                            output_elements * sizeof(float));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "SeedVR inference: " << ms << " ms total, " << ms / frames << " ms/image (" << frames << " images)"
                  << std::endl;
    }

private:
    cudaStream_t Stream()
    {
        return reinterpret_cast<cudaStream_t>(compute_stream_.GetHandle());
    }
    std::array<int64_t, 3> PublicShape(ImageRole role) const
    {
        if (role == ImageRole::Input)
            return {3, static_cast<int64_t>(shape_.source_height), static_cast<int64_t>(shape_.source_width)};
        if (role != ImageRole::Output)
            throw std::invalid_argument("Invalid image role");
        return {3, static_cast<int64_t>(shape_.target_height / 2 * 2),
                static_cast<int64_t>(shape_.target_width / 2 * 2)};
    }
    static bool IsCpu(const Ort::Value& v)
    {
        return v.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
    }
    void ValidateTensor(const Ort::Value& v, ImageRole role) const
    {
        if (!v || !v.IsTensor())
            throw std::invalid_argument("SeedVR expects a tensor");
        const auto info = v.GetTensorTypeAndShapeInfo();
        std::array<int64_t, 3> actual{};
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetDimensionsCount() != 3)
            throw std::invalid_argument("SeedVR expects contiguous float32 CHW");
        info.GetDimensions(actual.data(), actual.size());
        if (actual != PublicShape(role))
            throw std::invalid_argument("SeedVR tensor shape differs from construction shape");
        const auto memory = v.GetTensorMemoryInfo();
        if (memory.GetDeviceType() == OrtMemoryInfoDeviceType_CPU)
            return;
        const auto name = memory.GetAllocatorName();
        const bool cuda_memory = name == "Cuda" || name == sessions_->dit->DeviceMemory().GetAllocatorName();
        if (memory.GetDeviceType() != OrtMemoryInfoDeviceType_GPU || !cuda_memory ||
            memory.GetDeviceMemoryType() != OrtDeviceMemoryType_DEFAULT || memory.GetDeviceId() != cuda_device)
            throw std::invalid_argument("SeedVR supports CPU tensors or CUDA tensors on its configured device; got " +
                                        name + " device " + std::to_string(memory.GetDeviceId()) +
                                        ", expected CUDA device " + std::to_string(cuda_device));
    }
    struct SessionBundle
    {
        size_t padded_height = 0, padded_width = 0;
        std::unique_ptr<common::OrtRunner> encoder, dit, decoder;
#ifdef DIN_SEEDVR_NATIVE_VAE
        std::unique_ptr<NativeVae> native;
#endif
    };
    ModelConfig config_;
    DynamicOptions options_;
    ImageShape shape_;
    size_t frames, padded_height = 0, padded_width = 0, spatial_size = 0, noise_count = 0;
    int64_t latent_frames = 0, latent_height = 0, latent_width = 0, encoder_height = 0, encoder_width = 0,
            decoder_height = 0, decoder_width = 0;
    bool video = false, native_vae = false;
    int cuda_device = 0;
    // Declared before the stream, sessions and buffers so it outlives them.
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "din_seedvr2"};
    Ort::SyncStream compute_stream_{nullptr};
    std::unique_ptr<SessionBundle> sessions_;
    std::unique_ptr<common::TensorBuffer<Fp16>> pixels_, encoded_, condition_, latent_, velocity_, context_, timestep_;
    Ort::Value noise_{nullptr}, noise_host_{nullptr}, output_device_{nullptr}, output_host_{nullptr};
    Ort::Value input_device_{nullptr}, reference_{nullptr};
    Ort::Value lab_scratch_{nullptr};
    size_t lab_scratch_bytes_ = 0;
    std::unique_ptr<Ort::IoBinding> dit_io_;
    std::unique_ptr<MetadataBuffers> normal_, shifted_;
    std::unique_ptr<VaeExecution> encoder_exec_, decoder_exec_;
    Ort::RunOptions run_options_;
    std::string encoder_label_, dit_label_, decoder_label_, clip_label_;
};
}  // namespace detail

ImagePipeline::ImagePipeline(ModelConfig c, ImageShape s, DynamicOptions o)
    : core_(std::make_unique<detail::PipelineCore>(std::move(c), s, 1, std::move(o)))
{
}
ImagePipeline::~ImagePipeline() = default;
Ort::Value ImagePipeline::CreateImage(ImageRole r, ImageDevice d)
{
    return core_->CreateImage(r, d);
}
void ImagePipeline::RunImage(const Ort::Value& i, Ort::Value& o)
{
    core_->Run({&i, 1}, {&o, 1});
}
void ImagePipeline::UpdateOptions(const DynamicOptionsUpdate& u)
{
    core_->UpdateOptions(u);
}
VideoPipeline::VideoPipeline(ModelConfig c, VideoShape s, DynamicOptions o)
{
    if (s.frames < 5 || (s.frames - 1) % 4)
        throw std::invalid_argument("Video T must be 4n+1 with n >= 1");
    core_ = std::make_unique<detail::PipelineCore>(std::move(c), s.image, s.frames, std::move(o));
}
VideoPipeline::~VideoPipeline() = default;
Ort::Value VideoPipeline::CreateImage(ImageRole r, ImageDevice d)
{
    return core_->CreateImage(r, d);
}
void VideoPipeline::RunChunk(std::span<const Ort::Value> i, std::span<Ort::Value> o)
{
    core_->Run(i, o);
}
void VideoPipeline::UpdateOptions(const DynamicOptionsUpdate& u)
{
    core_->UpdateOptions(u);
}
}  // namespace din::seedvr
