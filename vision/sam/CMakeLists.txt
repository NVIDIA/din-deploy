# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared inference runtime and family-specific CUDA kernels.
enable_language(CUDA)
foreach (target IN ITEMS din_sam2_cuda din_sam3_cuda din_sam_kernels)
    string(REPLACE "din_" "" source "${target}")
    add_library(${target} STATIC cpp/${source}.cu)
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    target_link_libraries(${target} PUBLIC CUDA::cudart)
    set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
    )
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endforeach ()
if (DIN_ENABLE_NVTX)
    target_link_libraries(din_sam3_cuda PRIVATE nvtx3-cpp)
    target_compile_definitions(din_sam3_cuda PRIVATE DIN_ENABLE_NVTX=1)
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

add_library(din_sam_pipeline STATIC cpp/sam.cpp)
target_include_directories(din_sam_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(din_sam_pipeline PUBLIC din_common_ort
        PRIVATE din_sam_kernels din_sam3_cuda din_sam3_tokenizer nlohmann_json::nlohmann_json CUDA::cudart)

# Image/prompt loading and result serialization do not depend on the adapters.
add_library(din_sam_io STATIC cpp/sam2_io.cpp cpp/sam3_io.cpp)
target_include_directories(din_sam_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(din_sam_io PUBLIC din_common_ort din_common_io PRIVATE nlohmann_json::nlohmann_json)

# Compatibility adapters retain their device rendering and temporal-cache APIs.
add_library(din_sam2_pipeline STATIC cpp/sam2_pipeline.cpp)
target_link_libraries(din_sam2_pipeline PUBLIC din_sam_pipeline din_sam_io din_sam2_cuda
        nlohmann_json::nlohmann_json)
add_library(din_sam3_pipeline STATIC cpp/sam3_pipeline.cpp cpp/sam3_gpu.cpp)
target_link_libraries(din_sam3_pipeline PUBLIC din_sam_pipeline din_sam_io din_sam3_tokenizer
        PRIVATE din_sam3_cuda din_sam2_cuda nlohmann_json::nlohmann_json)
foreach (target IN ITEMS din_sam_pipeline din_sam_io din_sam2_pipeline din_sam3_pipeline)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach ()

add_din_executable(din_sam_cli cpp/sam_main.cpp)
target_link_libraries(din_sam_cli PRIVATE din_sam_pipeline din_sam_io argparse nlohmann_json::nlohmann_json)

add_din_executable(din_sam2_verify_masks cpp/sam2_verify_masks.cpp)
target_link_libraries(din_sam2_verify_masks PRIVATE din_sam_io)
