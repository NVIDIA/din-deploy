# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(artifacts_root "$ENV{DIN_DEPLOY_ARTIFACTS_ROOT}")
if(artifacts_root STREQUAL "")
    set(artifacts_root "${DEFAULT_ARTIFACTS_ROOT}")
endif()
if(artifacts_root STREQUAL "")
    message(FATAL_ERROR "DIN_DEPLOY_ARTIFACTS_ROOT is not set")
endif()

set(sam2_artifacts "${artifacts_root}/sam2")

if(TEST_KIND STREQUAL "image")
    execute_process(
        COMMAND "${DIN_SAM2_EXE}"
            --model-dir "${sam2_artifacts}/onnx"
            --provider trt-rtx
            --ep-cache "${EP_CACHE_DIR}"
            --ep-context-dir "${EP_CONTEXT_DIR}"
            --dump-masks
            "${sam2_artifacts}/frames/frame_000000.png"
            "${sam2_artifacts}/prompts/center_point.json"
            "${OUTPUT_DIR}/image_trt_rtx"
        RESULT_VARIABLE result
    )
elseif(TEST_KIND STREQUAL "image_verify")
    execute_process(
        COMMAND "${VERIFY_EXE}"
            --candidate "${OUTPUT_DIR}/image_trt_rtx"
            --reference "${sam2_artifacts}/reference/image"
            --min-iou 0.95
        RESULT_VARIABLE result
    )
elseif(TEST_KIND STREQUAL "video")
    execute_process(
        COMMAND "${DIN_SAM2_EXE}"
            --model-dir "${sam2_artifacts}/onnx"
            --provider trt-rtx
            --ep-cache "${EP_CACHE_DIR}"
            --ep-context-dir "${EP_CONTEXT_DIR}"
            --propagate
            --max-frames 50
            --dump-masks
            "${sam2_artifacts}/frames"
            "${sam2_artifacts}/prompts/center_point.json"
            "${OUTPUT_DIR}/video_trt_rtx"
        RESULT_VARIABLE result
    )
elseif(TEST_KIND STREQUAL "video_verify")
    execute_process(
        COMMAND "${VERIFY_EXE}"
            --candidate "${OUTPUT_DIR}/video_trt_rtx"
            --reference "${sam2_artifacts}/reference/video"
            --min-iou 0.95
        RESULT_VARIABLE result
    )
else()
    message(FATAL_ERROR "Unknown SAM2 test kind: ${TEST_KIND}")
endif()

if(NOT result EQUAL 0)
    message(FATAL_ERROR "SAM2 test failed with exit code ${result}")
endif()
