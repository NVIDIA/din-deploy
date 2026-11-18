// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include "sam_precision.h"
#include <cstdint>
#include "../../common/device_frame.h"

namespace din::sam3
{
    struct MaskRasterInputs
    {
        const float* probabilities[16]{};
        uint32_t ids[16]{};
        int count = 0;
    };

    void RasterizeMasks(MaskRasterInputs inputs, size_t pixels, uint32_t* labels,
                        uint8_t* combined, uint8_t* individual, void* stream);
    // XYXY bounds at probability > 0.5, followed by an invalid-probability flag.
    void MaskBounds(const float* probabilities, int width, int height, float* bounds, void* stream);
    // Top-down RGB8 input. Scratch is [height,1008,3], output [1,3,1008,1008].
    void PreprocessDeviceRgb(const uint8_t* input, int width, int height, std::ptrdiff_t row_bytes,
                             float* scratch, void* output, void* stream,
                             sam::detail::ModelPrecision precision = sam::detail::ModelPrecision::Float32);
    void ResizeMaskProbability(const void* logits, int mask_width, int mask_height,
                               void* probabilities, int width, int height, void* stream, bool binary = false,
                               sam::detail::ModelPrecision precision = sam::detail::ModelPrecision::Float32);
    // Consolidate the decoded 16-slot 288-square seed logits at native 1008 resolution.
    // Invalid slots do not compete; ties select the first active object, as in PyTorch.
    void PrepareSeedMemoryMasks(const void* logits, const void* valid, void* consolidated, void* stream,
                                sam::detail::ModelPrecision precision = sam::detail::ModelPrecision::Float32);
    void PackTemporalFeatures(const float* source, float* destination, const float* temporal, void* stream);
    // Select in query order, keeping IDs consistent with the native image API.
    void SelectQueries(const void* classes, const void* presence, int queries, float threshold,
                       int* selected, void* valid, int* error, void* stream,
                       sam::detail::ModelPrecision precision = sam::detail::ModelPrecision::Float32);
    void GatherMasks(const void* masks, const int* selected, void* packed, void* stream,
                     sam::detail::ModelPrecision precision = sam::detail::ModelPrecision::Float32);
    void MaskPointers(float* pointers, const float* valid, void* stream);
    void PointerPosition(float distance, const float* weights, const float* bias, float* output, void* stream);
    void WriteRgbaMasks(const vision::DeviceRgbaView& source, const vision::DeviceOutputView& output,
                        const float* masks, const float* valid, const float* object_logits,
                        int selected_id, bool labels, void* stream);
}
