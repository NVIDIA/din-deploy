# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(din_deploy
    VERSION 0.1.0
    DESCRIPTION "DO INFERENCE NOW local inference samples"
    LANGUAGES CXX
)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_ARCHITECTURES "86;89;120f")

if (WIN32)
    add_compile_definitions(NOMINMAX=1)
endif ()

if(WIN32)
    set(_din_default_artifacts_root "S:/din_deploy_artifacts")
else()
    set(_din_default_artifacts_root "/mnt/share/din_deploy_artifacts")
endif()
set(DIN_DEPLOY_ARTIFACTS_ROOT "${_din_default_artifacts_root}" CACHE PATH "Root directory containing din_deploy_artifacts test assets and exported models")
set(DIN_CLI_TEST_EP_CACHE_DIR "${CMAKE_BINARY_DIR}/rt_cache" CACHE PATH "TensorRT RTX runtime cache directory used by CLI CTest tests")
set(DIN_CLI_TEST_EP_CONTEXT_DIR "${CMAKE_BINARY_DIR}/ep_context" CACHE PATH "TensorRT RTX EP context directory used by CLI CTest tests")
unset(_din_default_artifacts_root)

if(CMAKE_CONFIGURATION_TYPES)
    set(DIN_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin/$<CONFIG>" CACHE PATH "Directory for built executables and runtime libraries")
else()
    set(DIN_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory for built executables and runtime libraries")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(onnxruntime)
include(nlohmann_json)
include(argparse)

find_package(CUDAToolkit REQUIRED)

include(tensorrt_rtx)

option(DIN_BUILD_TRT_RTX_EP "Build the TensorRT RTX execution provider from source" ON)
if(DIN_BUILD_TRT_RTX_EP)
    include(onnxruntime_trt_rtx_ep)
endif()

option(DIN_ENABLE_NVTX "Enable NVTX profiling" ON)
if(DIN_ENABLE_NVTX)
    include(nvtx)
endif()

function(add_din_executable target_name )
    add_executable(${target_name} ${ARGN})
    set_target_properties(${target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            LIBRARY_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
    )
    if (UNIX AND NOT APPLE)
        set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
    endif ()
    if (WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target_name}> ${DIN_EXTRA_RUNTIME_DLLS} $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
        )
    endif ()
endfunction()

function(add_din_shared_library target_name )
    add_library(${target_name} SHARED ${ARGN})
    set_target_properties(${target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            LIBRARY_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
            POSITION_INDEPENDENT_CODE ON
            WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endfunction()

add_subdirectory(common)
add_subdirectory(tests/base)
add_subdirectory(image_gen/flux2)
add_subdirectory(asr/rnnt)
add_subdirectory(vision/sam2)
add_subdirectory(asr/whisper)
add_subdirectory(asr/qwen3)
add_subdirectory(asr/diarization)

if(DIN_BUILD_TESTING)
    set(_din_cli_test_audio "${DIN_DEPLOY_ARTIFACTS_ROOT}/audio/de/schakaleundaraber_elli.mp3")

    function(add_din_cli_test test_name target_name)
        add_test(
            NAME "${test_name}"
            COMMAND "$<TARGET_FILE:${target_name}>" ${ARGN}
        )
        set_tests_properties("${test_name}" PROPERTIES
            LABELS "cli"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target_name}>"
        )
        if(UNIX AND NOT APPLE)
            if(ONNXRUNTIME_LIB_DIR)
                set(_din_cli_ld_path_modifications
                    "LD_LIBRARY_PATH=path_list_prepend:${ONNXRUNTIME_LIB_DIR}"
                )
            else ()
                set(_din_cli_ld_path_modifications
                        "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:${target_name}>"
                )
            endif()
            set_tests_properties("${test_name}" PROPERTIES
                ENVIRONMENT_MODIFICATION "${_din_cli_ld_path_modifications}"
            )
            unset(_din_cli_ld_path_modifications)
        endif()
    endfunction()

    # TODO add parakeet testing
    add_din_cli_test(din_asr_nemotron_cli din_asr_nemotron_cli
        "${_din_cli_test_audio}"
        --model-dir "${DIN_DEPLOY_ARTIFACTS_ROOT}/nemotron_asr35_chunk7"
        --provider trt-rtx
        --ep-cache "${DIN_CLI_TEST_EP_CACHE_DIR}"
        --ep-context-dir "${DIN_CLI_TEST_EP_CONTEXT_DIR}"
        --lang-id de-DE
    )

    function(add_din_whisper_cli_test test_name model_dir)
        add_din_cli_test("${test_name}" din_asr_whisper_cli
                --model-dir "${model_dir}"
                --provider trt-rtx
                --ep-cache "${DIN_CLI_TEST_EP_CACHE_DIR}"
                --ep-context-dir "${DIN_CLI_TEST_EP_CONTEXT_DIR}"
                "${_din_cli_test_audio}"
        )
    endfunction()

    add_din_whisper_cli_test(
        din_asr_whisper_small_cli
        "${DIN_DEPLOY_ARTIFACTS_ROOT}/whisper-small-onnx-fp16"
    )
    add_din_whisper_cli_test(
        din_asr_whisper_turbo_cli
        "${DIN_DEPLOY_ARTIFACTS_ROOT}/whisper-large-v3-turbo-onnx-fp16"
    )

    function(add_din_flux2_cli_test test_name model_dir ep processing)
        add_din_cli_test("${test_name}_${ep}_${processing}" din_flux2_cli
                --model-dir "${model_dir}"
                --provider "${ep}"
                --processing "${processing}"
                --ep-cache "${DIN_CLI_TEST_EP_CACHE_DIR}"
                --ep-context-dir "${DIN_CLI_TEST_EP_CONTEXT_DIR}"
                --num-images 3
                --output "${PROJECT_SOURCE_DIR}/${test_name}_${ep}_${processing}"
        )
    endfunction()

# TODO no bf16 support
#    add_din_flux2_cli_test(
#            din_flux2_bf16
#            "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-onnx"
#            cpu
#            cpu
#    )
    add_din_flux2_cli_test(
            din_flux2_fp8
            "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-fp8-onnx"
            trt-rtx
            cuda
    )
    add_din_flux2_cli_test(
            din_flux2_bf16
            "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-onnx"
            trt-rtx
            cuda
    )
    if (WIN32)
        add_din_flux2_cli_test(
                din_flux2_bf16
                "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-onnx"
                trt-rtx
                dx
        )
        add_din_flux2_cli_test(
                din_flux2_bf16
                "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-onnx"
                trt-rtx
                dx-cig
        )
    endif ()
    if (NOT WIN32)
        # TODO this test should be cross platform but experiences a device lost error on windows
        add_din_flux2_cli_test(
                din_flux2_bf16
                "${DIN_DEPLOY_ARTIFACTS_ROOT}/FLUX.2-klein-4B-onnx"
                trt-rtx
                vk
        )
    endif ()

    unset(_din_cli_test_audio)
endif()
