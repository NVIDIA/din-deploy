# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(random123
    GIT_REPOSITORY https://github.com/DEShawResearch/random123.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(random123)

# Random123 v1.14.0's MSVC feature header rejects ARM64 and otherwise selects
# x86 intrinsics for newer MSVC versions. Philox4x32 uses the scalar path.
if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64)$")
    set(_din_r123_msvc_features "${random123_SOURCE_DIR}/include/Random123/features/msvcfeatures.h")
    file(READ "${_din_r123_msvc_features}" _din_r123_features)
    set(_din_r123_old_guard "#if !defined(_M_IX86) && !defined(_M_X64)")
    set(_din_r123_new_guard "#if !defined(_M_IX86) && !defined(_M_X64) && !defined(_M_ARM64)")
    if(NOT _din_r123_features MATCHES "_M_ARM64")
        if(NOT _din_r123_features MATCHES "#if !defined\\(_M_IX86\\) && !defined\\(_M_X64\\)")
            message(FATAL_ERROR "Random123 MSVC architecture guard changed; review the ARM64 patch")
        endif()
        string(REPLACE "${_din_r123_old_guard}" "${_din_r123_new_guard}"
            _din_r123_features "${_din_r123_features}")
        file(WRITE "${_din_r123_msvc_features}" "${_din_r123_features}")
    endif()
    unset(_din_r123_msvc_features)
    unset(_din_r123_features)
    unset(_din_r123_old_guard)
    unset(_din_r123_new_guard)
endif()

add_library(din_random123 INTERFACE)
add_library(Random123::Random123 ALIAS din_random123)
target_include_directories(din_random123 SYSTEM INTERFACE "${random123_SOURCE_DIR}/include")
if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64)$")
    target_compile_definitions(din_random123 INTERFACE
        R123_USE_SSE=0
        R123_USE_SSE4_1=0
        R123_USE_SSE4_2=0
        R123_USE_EMMINTRIN_H=0
        R123_USE_SMMINTRIN_H=0
        R123_USE_WMMINTRIN_H=0
        R123_USE_CPUID_MSVC=0
    )
endif()
