# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)
include(FetchContent)
FetchContent_Declare(din_miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG f40cf03f80cdb7e741d43e53b7e706e8c1394bcf
    SOURCE_SUBDIR header-only)
FetchContent_MakeAvailable(din_miniaudio)

# Shared by CLI models and the standalone UI, without inference or windowing dependencies.
add_library(din_audio STATIC ${CMAKE_CURRENT_LIST_DIR}/../common/io/audio.cpp)
target_include_directories(din_audio PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../common/io
    ${din_miniaudio_SOURCE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(din_audio PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX)
    target_link_libraries(din_audio PRIVATE m)
endif()
set_target_properties(din_audio PROPERTIES POSITION_INDEPENDENT_CODE ON)
