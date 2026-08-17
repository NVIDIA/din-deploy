# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dr_libs)

if(NOT TARGET dr_libs::dr_libs)
    add_library(dr_libs::dr_libs INTERFACE IMPORTED)
    target_include_directories(dr_libs::dr_libs INTERFACE "${dr_libs_SOURCE_DIR}")
endif()

