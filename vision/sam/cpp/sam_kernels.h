// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sam.h"

namespace din::sam::detail {
constexpr size_t Feature = 256 * 72 * 72;
constexpr size_t Sam3Entry = 4 * Feature + 16 * 256;
struct TemporalMeta {
    uint64_t step = 0, length = 1;
    uint64_t frames[16]{};
    int initialized = 0, history_count = 0, next_slot = 1, error = 0;
    float valid[16]{};
};
struct Decision { int commit, seed, slot; };
void ConvertFrame(const FrameView&, uint8_t* rgb, void* stream);
void PreprocessSam2(const uint8_t*, int width, int height, void* image, bool half, void* stream);
void PackSam3(const TemporalMeta*, const float* ring, const float* embeddings,
              const float* weights, const float* bias, float* mi, float* mip, float* mem, float* mp,
              float* pointers, float* positions, bool* memory_valid, bool* pointer_valid,
              float* object_valid, void* stream);
void CommitSam3(TemporalMeta*, float* ring, Decision*, const float* image, const float* position,
                const float* seed_memory, const float* seed_position, const float* seed_pointers,
                const float* seed_valid, const float* seed_masks, const float* seed_logits,
                const float* track_memory, const float* track_position, const float* track_pointers,
                const float* track_masks, const float* track_logits,
                float* masks, float* valid, float* scores, float* logits, void* stream);
void DetectionScores(const float* classes, const float* presence, const int* selected,
                     const float* valid, float* scores, float* logits, const float* boxes,
                     float* output_boxes, int width, int height, void* stream);
void Sam2Result(const void* logits, const void* presence, bool half, float* masks,
                float* valid, float* scores, float* output_logits, void* stream);
void MemoryValidity(bool* valid, int active, void* stream);
void ResetMeta(TemporalMeta*, uint64_t length, void* stream);
void WriteMasks(const float* masks, const float* valid, const float* scores, const float* logits, const float* boxes, int objects,
                int mask_size, const MaskOutputView&, void* stream);
} // namespace din::sam::detail
