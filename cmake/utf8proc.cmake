# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)
include(FetchContent)

set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(utf8proc
    GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
    GIT_TAG v2.11.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(utf8proc)
set_target_properties(utf8proc PROPERTIES POSITION_INDEPENDENT_CODE ON)
