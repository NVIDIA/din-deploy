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
