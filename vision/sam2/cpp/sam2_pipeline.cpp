// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sam2_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "nvtx_helper.h"
#include "sam2_io.h"
#include <onnxruntime_run_options_config_keys.h>

namespace din::sam2
{
namespace
{

namespace fs = std::filesystem;

ort::ModelProfile ImagePreprocessProfile(const Sam2Config& cfg)
{
    return {
        .min_shapes = "image_rgb:1x1x1x3",
        .opt_shapes = "image_rgb:1x" + std::to_string(cfg.image_size) + "x" + std::to_string(cfg.image_size) + "x3",
        .max_shapes =
            "image_rgb:1x" + std::to_string(cfg.max_input_size) + "x" + std::to_string(cfg.max_input_size) + "x3",
        .cache_subpath = "image_preprocess",
    };
}

ort::ModelProfile MaskPostprocessProfile(const Sam2Config& cfg)
{
    const auto low_res_size = std::to_string(cfg.low_res_mask_size());
    const auto image_size = std::to_string(cfg.image_size);
    const auto max_input_size = std::to_string(cfg.max_input_size);
    const auto masks = "low_res_masks:1x1x" + low_res_size + "x" + low_res_size;
    return {
        .min_shapes = masks + ",output_size:1x1",
        .opt_shapes = masks + ",output_size:" + image_size + "x" + image_size,
        .max_shapes = masks + ",output_size:" + max_input_size + "x" + max_input_size,
        .cache_subpath = "mask_postprocess",
    };
}

}  // namespace

Sam2Pipeline::Sam2Pipeline(Sam2PipelineConfig config)
    : config_(std::move(config))
    , env_(ORT_LOGGING_LEVEL_WARNING, "din_sam2")
{
    DIN_NVTX_FUNC_RANGE();
    if (config_.provider == "trt-rtx")
    {
        ort::RegisterTensorRTRTXProvider(env_);
        compute_stream_ = ort::CreateTensorRTRTXComputeStream(env_);
    }
    model_config_ = LoadSam2Config(config_.model_dir / "metadata.json");
    const auto ep_context = ort::EpContextOptions{.output_dir = config_.ep_context_dir.string()};
    auto* compute_stream = config_.provider == "trt-rtx" ? &compute_stream_ : nullptr;

    const fs::path image_preprocess_path = config_.model_dir / "image_preprocess.onnx";
    if (!fs::is_regular_file(image_preprocess_path))
    {
        throw std::runtime_error("missing ONNX model: " + image_preprocess_path.string());
    }

    image_preprocess_ = std::make_unique<ort::OrtRunner>(env_, image_preprocess_path.string(), config_.provider,
                                                         config_.ep_cache_dir.string(), ep_context,
                                                         ImagePreprocessProfile(model_config_), compute_stream);
    encoder_ = std::make_unique<ort::OrtRunner>(env_, (config_.model_dir / "image_encoder.onnx").string(),
                                                config_.provider, config_.ep_cache_dir.string(), ep_context,
                                                ort::ModelProfile{.cache_subpath = "image_encoder"}, compute_stream);
    decoder_prompt_ = std::make_unique<ort::OrtRunner>(
        env_, (config_.model_dir / "sam2_decoder.onnx").string(), config_.provider, config_.ep_cache_dir.string(),
        ep_context, ort::ModelProfile{.cache_subpath = "sam2_decoder"}, compute_stream);

    // Video Decoder
    if (config_.propagate)
    {
        decoder_propagate_ = std::make_unique<ort::OrtRunner>(
            env_, (config_.model_dir / "sam2_decoder_propagate.onnx").string(), config_.provider,
            config_.ep_cache_dir.string(), ep_context, ort::ModelProfile{.cache_subpath = "sam2_decoder_propagate"},
            compute_stream);
        attention_ =
            std::make_unique<ort::OrtRunner>(env_, (config_.model_dir / "memory_attention.onnx").string(),
                                             config_.provider, config_.ep_cache_dir.string(), ep_context,
                                             ort::ModelProfile{.cache_subpath = "memory_attention"}, compute_stream);
        memory_encoder_ = std::make_unique<ort::OrtRunner>(
            env_, (config_.model_dir / "memory_encoder.onnx").string(), config_.provider, config_.ep_cache_dir.string(),
            ep_context, ort::ModelProfile{.cache_subpath = "memory_encoder"}, compute_stream);
    }
    const fs::path mask_postprocess_path = config_.model_dir / "mask_postprocess.onnx";
    if (!fs::is_regular_file(mask_postprocess_path))
    {
        throw std::runtime_error("missing ONNX model: " + mask_postprocess_path.string());
    }
    mask_postprocess_ = std::make_unique<ort::OrtRunner>(env_, mask_postprocess_path.string(), config_.provider,
                                                         config_.ep_cache_dir.string(), ep_context,
                                                         MaskPostprocessProfile(model_config_), compute_stream);

    use_device_io_ = encoder_->HasDeviceIo();
    if (use_device_io_)
    {
        run_options_.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    }
    AllocateBuffers();
}

void Sam2Pipeline::AllocateBuffers()
{
    image_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{1, 3, model_config_.image_size, model_config_.image_size}, use_device_io_);
    image_feature_0_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, model_config_.decoder_optimization_shapes.image_features_0, use_device_io_);
    image_feature_1_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, model_config_.decoder_optimization_shapes.image_features_1, use_device_io_);
    image_embeddings_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_,
        std::vector<int64_t>{1, model_config_.hidden_dim, model_config_.embedding_size(),
                             model_config_.embedding_size()},
        use_device_io_);
    vision_feat_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{model_config_.tokens(), 1, model_config_.hidden_dim}, use_device_io_);
    vision_pos_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{model_config_.tokens(), 1, model_config_.hidden_dim}, use_device_io_);
    input_masks_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{1, 1, model_config_.low_res_mask_size(), model_config_.low_res_mask_size()},
        use_device_io_);
    has_input_masks_ =
        std::make_unique<ort::TensorBuffer<ort::Fp16>>(*encoder_, std::vector<int64_t>{1, 1, 1, 1}, use_device_io_);
    mask_from_pts_true_ =
        std::make_unique<ort::TensorBuffer<ort::Fp16>>(*encoder_, std::vector<int64_t>{}, use_device_io_);
    mask_from_pts_false_ =
        std::make_unique<ort::TensorBuffer<ort::Fp16>>(*encoder_, std::vector<int64_t>{}, use_device_io_);

    input_masks_->Fill(Ort::Float16_t(0.0f));
    has_input_masks_->Fill(Ort::Float16_t(0.0f));
    mask_from_pts_true_->HostData()[0] = Ort::Float16_t(1.0f);
    mask_from_pts_false_->HostData()[0] = Ort::Float16_t(0.0f);
    input_masks_->CopyAsyncToDevice();
    has_input_masks_->CopyAsyncToDevice();
    mask_from_pts_true_->CopyAsyncToDevice();
    mask_from_pts_false_->CopyAsyncToDevice();

    prompt_coords_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{1, model_config_.max_points, 2}, use_device_io_);
    prompt_labels_ = std::make_unique<ort::TensorBuffer<int32_t>>(
        *encoder_, std::vector<int64_t>{1, model_config_.max_points}, use_device_io_);
    empty_coords_ =
        std::make_unique<ort::TensorBuffer<ort::Fp16>>(*encoder_, std::vector<int64_t>{1, 1, 2}, use_device_io_);
    empty_labels_ = std::make_unique<ort::TensorBuffer<int32_t>>(*encoder_, std::vector<int64_t>{1, 1}, use_device_io_);
    conditioned_embeddings_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{model_config_.tokens(), 1, model_config_.hidden_dim}, use_device_io_);
    masks_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{1, 1, model_config_.image_size, model_config_.image_size}, use_device_io_);
    low_res_masks_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
        *encoder_, std::vector<int64_t>{1, 1, model_config_.low_res_mask_size(), model_config_.low_res_mask_size()},
        use_device_io_);
    object_score_logits_ =
        std::make_unique<ort::TensorBuffer<ort::Fp16>>(*encoder_, std::vector<int64_t>{1, 1}, use_device_io_);
    mask_output_size_ =
        std::make_unique<ort::TensorBuffer<int64_t>>(*mask_postprocess_, std::vector<int64_t>{2}, use_device_io_);

    if (config_.propagate)
    {
        const int64_t memory_tokens = model_config_.max_memory_frames * model_config_.tokens();
        memory_ring_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
            *encoder_, std::vector<int64_t>{memory_tokens, 1, model_config_.memory_dim}, use_device_io_);
        memory_pos_ring_ = std::make_unique<ort::TensorBuffer<ort::Fp16>>(
            *encoder_, std::vector<int64_t>{memory_tokens, 1, model_config_.memory_dim}, use_device_io_);
        memory_ring_->Fill(Ort::Float16_t(0.0f));
        memory_pos_ring_->Fill(Ort::Float16_t(0.0f));
        memory_ring_->CopyAsyncToDevice();
        memory_pos_ring_->CopyAsyncToDevice();

        const size_t slot_bytes =
            static_cast<size_t>(model_config_.tokens() * model_config_.memory_dim) * sizeof(uint16_t);
        const std::vector<int64_t> slot_shape{model_config_.tokens(), 1, model_config_.memory_dim};
        memory_feature_slots_.reserve(static_cast<size_t>(model_config_.max_memory_frames));
        memory_pos_slots_.reserve(static_cast<size_t>(model_config_.max_memory_frames));
        memory_encoder_prompt_bindings_.reserve(static_cast<size_t>(model_config_.max_memory_frames));
        memory_encoder_propagate_bindings_.reserve(static_cast<size_t>(model_config_.max_memory_frames));
        for (int64_t slot = 0; slot < model_config_.max_memory_frames; ++slot)
        {
            const size_t byte_offset = static_cast<size_t>(slot) * slot_bytes;
            memory_feature_slots_.push_back(ort::AliasTensorBytes(*memory_ring_, *encoder_, use_device_io_, byte_offset,
                                                                  slot_bytes, slot_shape,
                                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            memory_pos_slots_.push_back(ort::AliasTensorBytes(*memory_pos_ring_, *encoder_, use_device_io_, byte_offset,
                                                              slot_bytes, slot_shape,
                                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));

            auto prompt_binding = std::make_unique<Ort::IoBinding>(memory_encoder_->session);
            prompt_binding->BindInput("vision_feat", vision_feat_->BindingValue());
            prompt_binding->BindInput("pred_masks_high_res", masks_->BindingValue());
            prompt_binding->BindInput("object_score_logits", object_score_logits_->BindingValue());
            prompt_binding->BindInput("is_mask_from_pts", mask_from_pts_true_->BindingValue());
            prompt_binding->BindOutput("memory_features", memory_feature_slots_.back());
            prompt_binding->BindOutput("memory_pos", memory_pos_slots_.back());
            memory_encoder_prompt_bindings_.push_back(std::move(prompt_binding));

            auto propagate_binding = std::make_unique<Ort::IoBinding>(memory_encoder_->session);
            propagate_binding->BindInput("vision_feat", vision_feat_->BindingValue());
            propagate_binding->BindInput("pred_masks_high_res", masks_->BindingValue());
            propagate_binding->BindInput("object_score_logits", object_score_logits_->BindingValue());
            propagate_binding->BindInput("is_mask_from_pts", mask_from_pts_false_->BindingValue());
            propagate_binding->BindOutput("memory_features", memory_feature_slots_.back());
            propagate_binding->BindOutput("memory_pos", memory_pos_slots_.back());
            memory_encoder_propagate_bindings_.push_back(std::move(propagate_binding));
        }
    }

    encoder_binding_ = std::make_unique<Ort::IoBinding>(encoder_->session);
    encoder_binding_->BindInput("image", image_->BindingValue());
    encoder_binding_->BindOutput("image_features_0", image_feature_0_->BindingValue());
    encoder_binding_->BindOutput("image_features_1", image_feature_1_->BindingValue());
    encoder_binding_->BindOutput("image_embeddings", image_embeddings_->BindingValue());
    encoder_binding_->BindOutput("vision_feat", vision_feat_->BindingValue());
    encoder_binding_->BindOutput("vision_pos", vision_pos_->BindingValue());

    decoder_prompt_binding_ = std::make_unique<Ort::IoBinding>(decoder_prompt_->session);
    decoder_prompt_binding_->BindInput("image_features_0", image_feature_0_->BindingValue());
    decoder_prompt_binding_->BindInput("image_features_1", image_feature_1_->BindingValue());
    decoder_prompt_binding_->BindInput("image_embeddings", image_embeddings_->BindingValue());
    decoder_prompt_binding_->BindInput("point_coords", prompt_coords_->BindingValue());
    decoder_prompt_binding_->BindInput("point_labels", prompt_labels_->BindingValue());
    decoder_prompt_binding_->BindInput("input_masks", input_masks_->BindingValue());
    decoder_prompt_binding_->BindInput("has_input_masks", has_input_masks_->BindingValue());
    decoder_prompt_binding_->BindOutput("masks", masks_->BindingValue());
    decoder_prompt_binding_->BindOutput("low_res_masks", low_res_masks_->BindingValue());
    decoder_prompt_binding_->BindOutput("object_score_logits", object_score_logits_->BindingValue());

    if (config_.propagate)
    {
        decoder_propagate_binding_ = std::make_unique<Ort::IoBinding>(decoder_propagate_->session);
        decoder_propagate_binding_->BindInput("image_features_0", image_feature_0_->BindingValue());
        decoder_propagate_binding_->BindInput("image_features_1", image_feature_1_->BindingValue());
        decoder_propagate_binding_->BindInput("image_embeddings", conditioned_embeddings_->BindingValue());
        decoder_propagate_binding_->BindInput("point_coords", empty_coords_->BindingValue());
        decoder_propagate_binding_->BindInput("point_labels", empty_labels_->BindingValue());
        decoder_propagate_binding_->BindInput("input_masks", input_masks_->BindingValue());
        decoder_propagate_binding_->BindInput("has_input_masks", has_input_masks_->BindingValue());
        decoder_propagate_binding_->BindOutput("masks", masks_->BindingValue());
        decoder_propagate_binding_->BindOutput("low_res_masks", low_res_masks_->BindingValue());
        decoder_propagate_binding_->BindOutput("object_score_logits", object_score_logits_->BindingValue());

        memory_attention_binding_ = std::make_unique<Ort::IoBinding>(attention_->session);
        memory_attention_binding_->BindInput("vision_feat", vision_feat_->BindingValue());
        memory_attention_binding_->BindInput("vision_pos", vision_pos_->BindingValue());
        memory_attention_binding_->BindInput("memory", memory_ring_->BindingValue());
        memory_attention_binding_->BindInput("memory_pos", memory_pos_ring_->BindingValue());
        memory_attention_binding_->BindOutput("conditioned_image_embeddings", conditioned_embeddings_->BindingValue());
    }
}

void Sam2Pipeline::PreparePromptBuffers(const Prompt& prompt)
{
    prompt_coords_->Fill(Ort::Float16_t(0.0f));
    prompt_labels_->Fill(-1);
    if (prompt.labels.size() != static_cast<size_t>(model_config_.max_points))
    {
        throw std::runtime_error(
            "prompt token count does not match exported SAM2 decoder: prompt has " +
            std::to_string(prompt.labels.size()) + " tokens, metadata expects " +
            std::to_string(model_config_.max_points) +
            ". Re-export with --max-points equal to the number of box-corner and point tokens in the prompt.");
    }
    const size_t prompt_points = prompt.labels.size();
    std::transform(prompt.coords.begin(), prompt.coords.begin() + static_cast<std::ptrdiff_t>(prompt_points * 2),
                   prompt_coords_->HostData(),
                   [](float coord)
                   {
                       return Ort::Float16_t(coord);
                   });
    std::copy_n(prompt.labels.begin(), prompt_points, prompt_labels_->HostData());

    const auto empty = EmptyPropagatePrompt();
    std::transform(empty.coords.begin(), empty.coords.end(), empty_coords_->HostData(),
                   [](float coord)
                   {
                       return Ort::Float16_t(coord);
                   });
    std::copy(empty.labels.begin(), empty.labels.end(), empty_labels_->HostData());

    prompt_coords_->CopyAsyncToDevice();
    prompt_labels_->CopyAsyncToDevice();
    empty_coords_->CopyAsyncToDevice();
    empty_labels_->CopyAsyncToDevice();
}

void Sam2Pipeline::ResizeImageRgbBufferIfNeeded(const Image& image)
{
    if (image.width > static_cast<uint32_t>(model_config_.max_input_size) ||
        image.height > static_cast<uint32_t>(model_config_.max_input_size))
    {
        throw std::runtime_error(
            "input image " + std::to_string(image.width) + "x" + std::to_string(image.height) +
            " exceeds image_preprocess max_input_size=" + std::to_string(model_config_.max_input_size));
    }
    if (image_rgb_ && image_rgb_width_ == image.width && image_rgb_height_ == image.height)
    {
        return;
    }
    image_rgb_ = std::make_unique<ort::TensorBuffer<uint8_t>>(
        *image_preprocess_,
        std::vector<int64_t>{1, static_cast<int64_t>(image.height), static_cast<int64_t>(image.width), 3},
        use_device_io_);
    image_rgb_width_ = image.width;
    image_rgb_height_ = image.height;
    image_preprocess_binding_ = std::make_unique<Ort::IoBinding>(image_preprocess_->session);
    image_preprocess_binding_->BindInput("image_rgb", image_rgb_->BindingValue());
    image_preprocess_binding_->BindOutput("image", image_->BindingValue());
}

void Sam2Pipeline::ResizeMaskRgbBufferIfNeeded(const Image& image)
{
    if (mask_rgb_ && mask_rgb_width_ == image.width && mask_rgb_height_ == image.height)
    {
        return;
    }
    mask_rgb_ = std::make_unique<ort::TensorBuffer<uint8_t>>(
        *mask_postprocess_,
        std::vector<int64_t>{1, static_cast<int64_t>(image.height), static_cast<int64_t>(image.width), 3},
        use_device_io_);
    mask_rgb_width_ = image.width;
    mask_rgb_height_ = image.height;
    mask_postprocess_binding_ = std::make_unique<Ort::IoBinding>(mask_postprocess_->session);
    mask_postprocess_binding_->BindInput("low_res_masks", low_res_masks_->BindingValue());
    mask_postprocess_binding_->BindInput("output_size", mask_output_size_->BindingValue());
    mask_postprocess_binding_->BindOutput("mask_rgb", mask_rgb_->BindingValue());
}

Image Sam2Pipeline::SegmentImage(const Image& image, const Prompt& prompt)
{
    PreparePromptBuffers(prompt);
    RunEncoder(image);
    RunFastDecoder(true);
    RunMaskPostprocess(image);
    if (use_device_io_)
    {
        auto mask_host_ready = mask_rgb_->CopyAsyncToHostWithNotification();
        mask_host_ready.Sync();
    }

    Image mask;
    mask.width = image.width;
    mask.height = image.height;
    const auto bytes = static_cast<size_t>(mask.width) * mask.height * 3;
    mask.rgb.assign(mask_rgb_->HostData(), mask_rgb_->HostData() + bytes);
    return mask;
}

void Sam2Pipeline::RunBatch(const std::vector<fs::path>& frames, const Prompt& prompt, const fs::path& output_path)
{
    if (frames.empty())
    {
        throw std::runtime_error("no input frames found");
    }
    PreparePromptBuffers(prompt);

    const bool write_masks = config_.dump_masks;
    if (write_masks)
    {
        fs::create_directories(output_path);
    }

    detail::AsyncMaskWriter writer(write_masks);
    auto load_frame = [](const fs::path& path)
    {
        din::common::nvtx_scoped_range range{"sam2_async_load_png"};
        return LoadPng(path);
    };
    auto next_image = std::async(std::launch::async, load_frame, frames.front());
    for (size_t frame_idx = 0; frame_idx < frames.size(); ++frame_idx)
    {
        Image image = next_image.get();
        if (frame_idx + 1 < frames.size())
        {
            next_image = std::async(std::launch::async, load_frame, frames[frame_idx + 1]);
        }

        din::common::nvtx_scoped_range frame_range{"sam2_frame"};
        const auto mask_path = output_path / FrameMaskName(frame_idx, "_fast.png");
        RunFrame(image, mask_path, write_masks, &writer);
    }
    writer.Finish();

    std::cout << "Processed " << frames.size() << " frame" << (frames.size() == 1 ? "" : "s") << '\n';
}

void Sam2Pipeline::RunFrame(const Image& image, const fs::path& mask_path, bool write_mask,
                            detail::AsyncMaskWriter* writer)
{
    RunEncoder(image);

    const bool prompted = !config_.propagate || active_memory_ == 0;
    if (!prompted)
    {
        RunFastMemoryAttention();
    }
    RunFastDecoder(prompted);
    if (config_.propagate)
    {
        RunFastMemoryEncoder(prompted);
    }
    if (write_mask)
    {
        ort::NotificationPtr mask_host_ready{nullptr};
        {
            din::common::nvtx_scoped_range range{"sam2_mask_postprocess"};
            RunMaskPostprocess(image);
            mask_host_ready = mask_rgb_->CopyAsyncToHostWithNotification();
        }
        {
            din::common::nvtx_scoped_range range{"sam2_mask_rgb_wait"};
            mask_host_ready.Sync();
        }
        {
            din::common::nvtx_scoped_range range{"sam2_enqueue_rgb_mask"};
            writer->EnqueueRgb(mask_path, mask_rgb_->HostData(), image.width, image.height);
        }
        std::cout << "Wrote " << mask_path << '\n';
    }
}

void Sam2Pipeline::RunEncoder(const Image& image)
{
    ResizeImageRgbBufferIfNeeded(image);
    {
        din::common::nvtx_scoped_range range{"sam2_copy_rgb_to_pinned"};
        std::copy(image.rgb.begin(), image.rgb.end(), image_rgb_->HostData());
    }
    image_rgb_->CopyAsyncToDevice();

    {
        din::common::nvtx_scoped_range range{"sam2_model_image_preprocess"};
        image_preprocess_->session.Run(run_options_, *image_preprocess_binding_);
    }

    {
        din::common::nvtx_scoped_range range{"sam2_model_image_encoder"};
        encoder_->session.Run(run_options_, *encoder_binding_);
    }
}

void Sam2Pipeline::RunFastMemoryAttention()
{
    {
        din::common::nvtx_scoped_range range{"sam2_model_memory_attention"};
        attention_->session.Run(run_options_, *memory_attention_binding_);
    }
}

void Sam2Pipeline::RunFastDecoder(bool prompted)
{
    auto& binding = prompted ? *decoder_prompt_binding_ : *decoder_propagate_binding_;
    {
        din::common::nvtx_scoped_range range{prompted ? "sam2_model_decoder_prompt" : "sam2_model_decoder_propagate"};
        (prompted ? decoder_prompt_ : decoder_propagate_)->session.Run(run_options_, binding);
    }
}

void Sam2Pipeline::RunMaskPostprocess(const Image& image)
{
    ResizeMaskRgbBufferIfNeeded(image);
    auto* output_size = mask_output_size_->HostData();
    output_size[0] = static_cast<int64_t>(image.height);
    output_size[1] = static_cast<int64_t>(image.width);
    mask_output_size_->CopyAsyncToDevice();

    {
        din::common::nvtx_scoped_range range{"sam2_model_mask_postprocess"};
        mask_postprocess_->session.Run(run_options_, *mask_postprocess_binding_);
    }
}

void Sam2Pipeline::RunFastMemoryEncoder(bool prompted)
{
    auto& bindings = prompted ? memory_encoder_prompt_bindings_ : memory_encoder_propagate_bindings_;
    auto& binding = *bindings[next_slot_];
    {
        din::common::nvtx_scoped_range range{"sam2_model_memory_encoder"};
        memory_encoder_->session.Run(run_options_, binding);
    }

    active_memory_ = std::min(active_memory_ + 1, static_cast<size_t>(model_config_.max_memory_frames));
    next_slot_ = (next_slot_ + 1) % static_cast<size_t>(model_config_.max_memory_frames);
}

}  // namespace din::sam2
