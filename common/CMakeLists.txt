# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(din_common_ort STATIC
    ort_session.cpp
)

target_include_directories(din_common_ort
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(din_common_ort
    PUBLIC
        onnxruntime_interface
    PRIVATE
        ${CMAKE_DL_LIBS}
)

if(TARGET onnxruntime_trt_rtx_ep_interface)
    target_link_libraries(din_common_ort PUBLIC onnxruntime_trt_rtx_ep_interface)
endif()

if(DIN_ENABLE_NVTX)
    target_link_libraries(din_common_ort PUBLIC nvtx3-cpp)
    target_compile_definitions(din_common_ort PUBLIC DIN_ENABLE_NVTX=1)
endif()

set_target_properties(din_common_ort PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_subdirectory(io)

# CPU callers do not depend on CUDA headers or libraries.
include(random123)
add_library(din_common_rng STATIC rng.cpp)
target_include_directories(din_common_rng PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(din_common_rng PUBLIC cxx_std_20)
target_link_libraries(din_common_rng PRIVATE Random123::Random123)
set_target_properties(din_common_rng PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
add_library(din_common_rng_cuda STATIC rng_cuda.cu)
target_include_directories(din_common_rng_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(din_common_rng_cuda PUBLIC CUDA::cudart)
set_target_properties(din_common_rng_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
