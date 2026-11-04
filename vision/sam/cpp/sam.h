// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace din::sam {
enum class ModelFamily { Sam2, Sam3 };
enum class MemoryLocation { Host, Device };
enum class FrameFormat { Rgb8, Rgba32F };
enum class Origin { TopLeft, BottomLeft };
enum class MaskFormat { Logits, Probabilities, Binary, Labels };
enum class ScalarType { Float32, Float16, UInt8, UInt32 };

struct FrameView {
    const void* data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t row_bytes = 0;
    MemoryLocation memory = MemoryLocation::Device;
    FrameFormat format = FrameFormat::Rgb8;
    Origin origin = Origin::TopLeft;
    bool premultiplied = false;
    int device = 0;
};
// Instance-major planes; zero strides mean tightly packed. Labels use one plane.
// selected_id=0 writes every instance; otherwise writes one selected plane.
// Metadata arrays, when supplied, have capacity MaxObjects() and share memory.
struct MaskOutputView {
    void* data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t row_bytes = 0, instance_bytes = 0;
    MemoryLocation memory = MemoryLocation::Device;
    MaskFormat format = MaskFormat::Logits;
    ScalarType type = ScalarType::Float32;
    int selected_id = 0;
    float* validity = nullptr;
    float* scores = nullptr;
    uint32_t* ids = nullptr;
    float* presence_logits = nullptr;
    // Optional SAM3 image-detector XYXY pixel boxes (objects*4); zero for tracking/SAM2.
    float* detection_boxes = nullptr;
    int device = 0;
};
struct SamConfig {
    ModelFamily model = ModelFamily::Sam3;
    std::filesystem::path model_dir;
    std::filesystem::path cache_dir;
    std::string provider = "trt-rtx";
    // Borrowed cudaStream_t. Null requests a private nonblocking stream.
    void* stream = nullptr;
    int device = 0;
    int max_source_width = 4096, max_source_height = 4096;
    size_t state_capacity = 1, completion_capacity = 8;
    float confidence_threshold = .5f;
    bool cuda_graph = true;
    // Optional explicit diagnostic mode; readback/file I/O is outside the fast-path contract.
    std::filesystem::path diagnostic_dir, diagnostic_graph_dir, diagnostic_decoder;
};
struct Prompt {
    std::string text;
    std::vector<float> coordinates;
    std::vector<int32_t> labels;
};
class PreparedPrompt {
public:
    PreparedPrompt() = default;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Sam;
};
struct Initialization {
    PreparedPrompt prompt;
    size_t sequence_length = 1;
    // false runs independent segmentation, without recurrent history updates.
    bool tracking = true;
};
class State {
public:
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    size_t Step() const;
private:
    struct Impl;
    explicit State(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class Sam;
    friend class StateLease;
};
class StateLease {
public:
    StateLease() = default;
    ~StateLease();
    StateLease(StateLease&&) noexcept;
    StateLease& operator=(StateLease&&) noexcept;
    StateLease(const StateLease&) = delete;
    StateLease& operator=(const StateLease&) = delete;
    State* get() const { return state_; }
    State& operator*() const { return *state_; }
    State* operator->() const { return state_; }
    explicit operator bool() const { return state_ != nullptr; }
private:
    State* state_ = nullptr;
    uint64_t generation_ = 0;
    friend class Sam;
};
class CompletionToken {
public:
    CompletionToken() = default;
    ~CompletionToken();
    CompletionToken(CompletionToken&&) noexcept;
    CompletionToken& operator=(CompletionToken&&) noexcept;
    CompletionToken(const CompletionToken&) = delete;
    CompletionToken& operator=(const CompletionToken&) = delete;
    bool Ready() const;
    void WaitOn(void* cuda_stream) const;
    void Wait() const;
private:
    struct Slot;
    Slot* slot_ = nullptr;
    friend class Sam;
};
struct FrameSubmission {
    State* state;
    CompletionToken completion;
};
// Calls are serialized. Sam must outlive its leases, prompts and tokens.
// Device RunFrame only enqueues; CPU and host-output adapters are synchronous.
class Sam {
public:
    explicit Sam(const SamConfig&);
    ~Sam();
    Sam(const Sam&) = delete;
    Sam& operator=(const Sam&) = delete;
    PreparedPrompt PreparePrompt(const Prompt&);
    void ReserveStates(size_t capacity);
    StateLease AcquireState();
    void ReleaseState(StateLease&&);
    FrameSubmission RunFrame(const FrameView&, const MaskOutputView&, State& destination,
                             const State* previous = nullptr, const Initialization* = nullptr);
    int MaxObjects() const;
    int MaskSize() const;
    size_t StateBytes() const;
    void* Stream() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace din::sam
