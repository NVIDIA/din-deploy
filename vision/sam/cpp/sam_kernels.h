// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sam.h"
#include "sam_precision.h"

namespace din::sam::detail
{
    constexpr size_t Feature = 256 * 72 * 72;
    constexpr size_t Sam3Entry = 4 * Feature + 16 * 256;

    struct TemporalMeta
    {
        uint64_t step = 0, length = 1;
        uint64_t frames[16]{};
        int initialized = 0, history_count = 0, next_slot = 1, error = 0;
        float valid[16]{};
    };

    struct Decision
    {
        int commit, seed, slot;
    };

    void ConvertFrame(const FrameView&, uint8_t* rgb, void* stream);
    void PreprocessSam2(const uint8_t*, int width, int height, void* image, ModelPrecision, void* stream);
    void PackSam3(const TemporalMeta*, const void* ring, const float* embeddings,
                  const float* weights, const float* bias, void* mi, void* mip, void* mem, void* mp,
                  void* pointers, void* positions, bool* memory_valid, bool* pointer_valid,
                  void* object_valid, ModelPrecision precision, void* stream);
    void CommitSam3(TemporalMeta*, void* ring, Decision*, const void* image, const void* position,
                    const void* seed_memory, const void* seed_position, const void* seed_pointers,
                    const void* seed_valid, const void* seed_masks, const void* seed_logits,
                    const void* track_memory, const void* track_position, const void* track_pointers,
                    const void* track_masks, const void* track_logits,
                    float* masks, float* valid, float* scores, float* logits, ModelPrecision precision, void* stream);
    void DetectionScores(const void* classes, const void* presence, const int* selected,
                         const void* valid, float* scores, float* logits, const void* boxes,
                         float* output_boxes, int width, int height, ModelPrecision precision, void* stream);
    void Sam2Result(const void* logits, const void* presence, ModelPrecision, float* masks,
                    float* valid, float* scores, float* output_logits, void* stream);
    void ModelToFloat(const void*, float*, size_t count, ModelPrecision, void* stream);
    void MemoryValidity(bool* valid, int active, void* stream);
    void ResetMeta(TemporalMeta*, uint64_t length, void* stream);
    void WriteMasks(const float* masks, const float* valid, const float* scores, const float* logits,
                    const float* boxes, int objects,
                    int mask_size, const MaskOutputView&, void* stream);
} // namespace din::sam::detail
