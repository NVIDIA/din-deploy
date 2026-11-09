# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
enable_language(CUDA)

# SAM2 video segmentation. Follows the asr/rnnt layering: shared ORT helpers,
# a processing library (din_sam2_pipeline), and a thin CLI.

add_library(din_sam2_cuda STATIC cpp/sam2_cuda.cu)
target_link_libraries(din_sam2_cuda PUBLIC CUDA::cudart)
set_target_properties(din_sam2_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
target_compile_options(din_sam2_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_library(din_sam2_pipeline STATIC
    cpp/sam2_io.cpp
    cpp/sam2_pipeline.cpp
)

target_include_directories(din_sam2_pipeline
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)

target_link_libraries(din_sam2_pipeline
    PUBLIC
        lodepng
        nlohmann_json::nlohmann_json
        din_common_ort
        din_common_io
        din_sam2_cuda
        din_sam_pipeline
)

set_target_properties(din_sam2_pipeline PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_din_executable(din_sam2_cli
    cpp/sam2_main.cpp
)

target_link_libraries(din_sam2_cli PRIVATE
    din_sam2_pipeline
    argparse
)

add_din_executable(din_sam2_verify_masks cpp/sam2_verify_masks.cpp)
target_link_libraries(din_sam2_verify_masks PRIVATE din_sam2_pipeline)

if(DIN_BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_sam2_ctest.cmake")
    set(_sam2_test_output "${CMAKE_BINARY_DIR}/sam2_tests")
    set(_sam2_ctest_runner "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_sam2_ctest.cmake")

    add_test(
        NAME din_sam2_image_trt_rtx
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=image
            -DDIN_SAM2_EXE=$<TARGET_FILE:din_sam2_cli>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DEP_CACHE_DIR=${DIN_CLI_TEST_EP_CACHE_DIR}
            -DEP_CONTEXT_DIR=${DIN_CLI_TEST_EP_CONTEXT_DIR}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    add_test(
        NAME din_sam2_image_trt_rtx_verify
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=image_verify
            -DVERIFY_EXE=$<TARGET_FILE:din_sam2_verify_masks>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    set_tests_properties(din_sam2_image_trt_rtx_verify PROPERTIES DEPENDS din_sam2_image_trt_rtx)

    add_test(
        NAME din_sam2_video_trt_rtx
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=video
            -DDIN_SAM2_EXE=$<TARGET_FILE:din_sam2_cli>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DEP_CACHE_DIR=${DIN_CLI_TEST_EP_CACHE_DIR}
            -DEP_CONTEXT_DIR=${DIN_CLI_TEST_EP_CONTEXT_DIR}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    add_test(
        NAME din_sam2_video_trt_rtx_verify
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=video_verify
            -DVERIFY_EXE=$<TARGET_FILE:din_sam2_verify_masks>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    set_tests_properties(din_sam2_video_trt_rtx_verify PROPERTIES DEPENDS din_sam2_video_trt_rtx)

    set_tests_properties(
        din_sam2_image_trt_rtx
        din_sam2_image_trt_rtx_verify
        din_sam2_video_trt_rtx
        din_sam2_video_trt_rtx_verify
        PROPERTIES LABELS "sam2;cli"
    )

    unset(_sam2_ctest_runner)
    unset(_sam2_test_output)
endif()

add_library(din_sam3_tokenizer STATIC cpp/sam3_tokenizer.cpp)
target_include_directories(din_sam3_tokenizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(din_sam3_tokenizer PRIVATE nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(din_sam3_tokenizer PRIVATE icu)
else()
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    target_link_libraries(din_sam3_tokenizer PRIVATE ICU::uc ICU::i18n)
endif()
add_library(din_sam3_cuda STATIC cpp/sam3_cuda.cu)
target_link_libraries(din_sam3_cuda PUBLIC CUDA::cudart)
if(DIN_ENABLE_NVTX)
    target_link_libraries(din_sam3_cuda PRIVATE nvtx3-cpp)
    target_compile_definitions(din_sam3_cuda PRIVATE DIN_ENABLE_NVTX=1)
endif()
set_target_properties(din_sam3_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
target_compile_options(din_sam3_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_library(din_sam3_pipeline STATIC cpp/sam3_pipeline.cpp cpp/sam3_gpu.cpp cpp/sam3_io.cpp)
target_include_directories(din_sam3_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp" PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(din_sam3_pipeline PUBLIC din_common_ort din_sam3_tokenizer din_sam_pipeline PRIVATE lodepng nlohmann_json::nlohmann_json din_sam3_cuda din_sam2_cuda)
add_din_executable(din_sam3_cli cpp/sam3_main.cpp)
target_link_libraries(din_sam3_cli PRIVATE din_sam3_pipeline din_sam2_pipeline argparse nlohmann_json::nlohmann_json)
if(DIN_BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/sam3_tokenizer_tests.cpp")
    add_executable(din_sam3_tokenizer_tests tests/sam3_tokenizer_tests.cpp)
    target_link_libraries(din_sam3_tokenizer_tests PRIVATE din_sam3_tokenizer nlohmann_json::nlohmann_json)
    add_test(NAME din_sam3_tokenizer COMMAND din_sam3_tokenizer_tests
        "${PROJECT_SOURCE_DIR}/artifacts/sam3/onnx-fp32/tokenizer.json"
        "${PROJECT_SOURCE_DIR}/artifacts/sam3/onnx-fp32/tokenizer_tests.json")
    set_tests_properties(din_sam3_tokenizer PROPERTIES SKIP_RETURN_CODE 77 LABELS "sam3")
endif()

add_library(din_sam_kernels STATIC cpp/sam_kernels.cu)
target_include_directories(din_sam_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(din_sam_kernels PUBLIC CUDA::cudart)
set_target_properties(din_sam_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
target_compile_options(din_sam_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_library(din_sam_pipeline STATIC cpp/sam.cpp)
target_include_directories(din_sam_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(din_sam_pipeline PUBLIC din_common_ort PRIVATE din_sam_kernels din_sam3_cuda din_sam3_tokenizer nlohmann_json::nlohmann_json CUDA::cudart)
set_target_properties(din_sam_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(DIN_BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/temporal_tests.cpp")
    add_executable(din_sam_temporal_tests tests/temporal_tests.cpp)
    target_link_libraries(din_sam_temporal_tests PRIVATE din_sam_kernels CUDA::cudart)
    add_test(NAME din_sam_temporal_cuda COMMAND din_sam_temporal_tests)
endif()
