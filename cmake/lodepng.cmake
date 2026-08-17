# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

### LodePNG
include(FetchContent)
set(LODEPNG_COMPILE_DECODER ON)
FetchContent_Declare(lodepng_fetch
        GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
)
FetchContent_MakeAvailable(lodepng_fetch)

add_library(lodepng STATIC
        ${lodepng_fetch_SOURCE_DIR}/lodepng.cpp
)
target_include_directories(lodepng PUBLIC
        ${lodepng_fetch_SOURCE_DIR}
)
set_target_properties(lodepng PROPERTIES POSITION_INDEPENDENT_CODE ON)
