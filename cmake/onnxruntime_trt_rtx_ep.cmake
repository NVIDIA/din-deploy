# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(ExternalProject)

set(ONNXRUNTIME_TRT_RTX_EP_GIT_REPOSITORY
    "https://github.com/NVIDIA/TensorRT-RTX-EP-ABI.git"
    CACHE STRING "Git repository for the ONNX Runtime TensorRT RTX Execution Provider")
set(ONNXRUNTIME_TRT_RTX_EP_GIT_TAG
    "v0.4.0"
    CACHE STRING "Git tag, branch, or commit for the ONNX Runtime TensorRT RTX Execution Provider")
set(ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_TOKEN
    ""
    CACHE STRING "Optional HTTP token for cloning the ONNX Runtime TensorRT RTX Execution Provider")
set(ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_USERNAME
    "oauth2"
    CACHE STRING "HTTP username used with ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_TOKEN")
option(ONNXRUNTIME_TRT_RTX_EP_BUILD_TESTS "Build ONNX Runtime TensorRT RTX Execution Provider tests" OFF)

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must be set before including onnxruntime_trt_rtx_ep.cmake")
endif()

if(NOT TRT_RTX_ROOT)
    message(FATAL_ERROR "TRT_RTX_ROOT must be set before including onnxruntime_trt_rtx_ep.cmake")
endif()

set(_onnxruntime_trt_rtx_ep_prefix "${CMAKE_BINARY_DIR}/_deps/onnxruntime_trt_rtx_ep")
set(_onnxruntime_trt_rtx_ep_source_dir "${_onnxruntime_trt_rtx_ep_prefix}/src")
set(_onnxruntime_trt_rtx_ep_binary_dir "${_onnxruntime_trt_rtx_ep_prefix}/build")

set(_onnxruntime_trt_rtx_ep_cmake_args
    "-DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
    "-DTRT_RTX_ROOT=${TRT_RTX_ROOT}"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}"
    "-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}"
    "-DBUILD_TESTS=${ONNXRUNTIME_TRT_RTX_EP_BUILD_TESTS}"
)

if(WIN32)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DUSE_PRECOMPILED_HOST_PROTOC=ON"
    )
endif()

if(CMAKE_BUILD_TYPE)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
endif()

if(CMAKE_MAKE_PROGRAM)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    )
endif()

if(CMAKE_C_COMPILER)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    )
endif()

if(CMAKE_CXX_COMPILER)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    )
endif()

if(CUDAToolkit_ROOT)
    list(APPEND _onnxruntime_trt_rtx_ep_cmake_args
        "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
    )
endif()

set(_onnxruntime_trt_rtx_ep_generator_args)
if(CMAKE_GENERATOR)
    list(APPEND _onnxruntime_trt_rtx_ep_generator_args
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    )
endif()
if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND _onnxruntime_trt_rtx_ep_generator_args
        CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}"
    )
endif()
if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND _onnxruntime_trt_rtx_ep_generator_args
        CMAKE_GENERATOR_TOOLSET "${CMAKE_GENERATOR_TOOLSET}"
    )
endif()

set(_onnxruntime_trt_rtx_ep_build_command
    "${CMAKE_COMMAND}" --build "<BINARY_DIR>"
)
if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND _onnxruntime_trt_rtx_ep_build_command --config "$<CONFIG>")
endif()

set(_onnxruntime_trt_rtx_ep_git_repository "${ONNXRUNTIME_TRT_RTX_EP_GIT_REPOSITORY}")
if(ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_TOKEN)
    if(NOT ONNXRUNTIME_TRT_RTX_EP_GIT_REPOSITORY MATCHES "^https://")
        message(FATAL_ERROR "ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_TOKEN requires an https:// Git repository URL")
    endif()
    string(REGEX REPLACE
        "^https://"
        "https://${ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_USERNAME}:${ONNXRUNTIME_TRT_RTX_EP_GIT_HTTP_TOKEN}@"
        _onnxruntime_trt_rtx_ep_git_repository
        "${ONNXRUNTIME_TRT_RTX_EP_GIT_REPOSITORY}")
endif()

ExternalProject_Add(onnxruntime_trt_rtx_ep_external
    GIT_REPOSITORY "${_onnxruntime_trt_rtx_ep_git_repository}"
    GIT_TAG "${ONNXRUNTIME_TRT_RTX_EP_GIT_TAG}"
    GIT_SHALLOW TRUE
    SOURCE_DIR "${_onnxruntime_trt_rtx_ep_source_dir}"
    BINARY_DIR "${_onnxruntime_trt_rtx_ep_binary_dir}"
    ${_onnxruntime_trt_rtx_ep_generator_args}
    CMAKE_ARGS ${_onnxruntime_trt_rtx_ep_cmake_args}
    BUILD_COMMAND ${_onnxruntime_trt_rtx_ep_build_command}
    INSTALL_COMMAND ""
    UPDATE_DISCONNECTED TRUE
)

add_custom_target(onnxruntime_trt_rtx_ep
    DEPENDS onnxruntime_trt_rtx_ep_external
)

if(WIN32)
    set(_onnxruntime_trt_rtx_ep_runtime_library_name "onnxruntime_providers_nv_tensorrt_rtx.dll")
else()
    set(_onnxruntime_trt_rtx_ep_runtime_library_name "libonnxruntime_providers_nv_tensorrt_rtx.so")
endif()

set(ONNXRUNTIME_TRT_RTX_EP_RUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime_trt_rtx_ep/$<CONFIG>")
set(ONNXRUNTIME_TRT_RTX_EP_RUNTIME_LIBRARY
    "${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_DIR}/${_onnxruntime_trt_rtx_ep_runtime_library_name}")

set(_onnxruntime_trt_rtx_ep_runtime_dependencies)
if(WIN32)
    list(APPEND _onnxruntime_trt_rtx_ep_runtime_dependencies ${TRTRTX_RUNTIME_DLLS})
else()
    if(TRTRTX_LIB)
        list(APPEND _onnxruntime_trt_rtx_ep_runtime_dependencies "${TRTRTX_LIB}")
    endif()
    if(TRTRTX_PARSER_LIB)
        list(APPEND _onnxruntime_trt_rtx_ep_runtime_dependencies "${TRTRTX_PARSER_LIB}")
    endif()
endif()
if(_onnxruntime_trt_rtx_ep_runtime_dependencies)
    list(REMOVE_DUPLICATES _onnxruntime_trt_rtx_ep_runtime_dependencies)
endif()


set(_onnxruntime_trt_rtx_ep_runtime_commands
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_DIR}"
)

list(APPEND _onnxruntime_trt_rtx_ep_runtime_commands
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_onnxruntime_trt_rtx_ep_binary_dir}/$<CONFIG>/${_onnxruntime_trt_rtx_ep_runtime_library_name}"
        "${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_DIR}"
)

foreach(_onnxruntime_trt_rtx_ep_runtime_dependency IN LISTS _onnxruntime_trt_rtx_ep_runtime_dependencies)
    list(APPEND _onnxruntime_trt_rtx_ep_runtime_commands
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_onnxruntime_trt_rtx_ep_runtime_dependency}"
                "${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_DIR}"
    )
endforeach()

add_custom_target(onnxruntime_trt_rtx_ep_runtime
    ${_onnxruntime_trt_rtx_ep_runtime_commands}
    DEPENDS onnxruntime_trt_rtx_ep_external onnxruntime_runtime
    VERBATIM
)

add_dependencies(onnxruntime_trt_rtx_ep onnxruntime_trt_rtx_ep_runtime)

add_library(onnxruntime_trt_rtx_ep_interface INTERFACE)
target_link_libraries(onnxruntime_trt_rtx_ep_interface INTERFACE
        onnxruntime_interface
)
target_compile_definitions(onnxruntime_trt_rtx_ep_interface INTERFACE
    ONNXRUNTIME_TRT_RTX_EP_LIBRARY_PATH=\"${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_LIBRARY}\"
)
message(STATUS "ORT TRT RTX EP will be located at ONNXRUNTIME_TRT_RTX_EP_LIBRARY_PATH=\"${ONNXRUNTIME_TRT_RTX_EP_RUNTIME_LIBRARY}\"")
add_dependencies(onnxruntime_trt_rtx_ep_interface onnxruntime_trt_rtx_ep_runtime)
