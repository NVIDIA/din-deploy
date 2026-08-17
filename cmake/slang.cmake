# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(SLANG_ROOT "" CACHE PATH "Optional path to Slang SDK root directory containing slangc")
set(SLANG_VERSION "2026.12" CACHE STRING "Slang version to download when slangc is not found")
set(SLANG_DOWNLOAD_BASE_URL
    "https://github.com/shader-slang/slang/releases/download"
    CACHE STRING "Base URL for Slang release downloads")

string(REGEX REPLACE "^v" "" _slang_version "${SLANG_VERSION}")

if(SLANG_ROOT AND NOT SLANGC)
    unset(SLANGC CACHE)
    find_program(SLANGC
        NAMES slangc slangc.exe
        HINTS "${SLANG_ROOT}" "${SLANG_ROOT}/bin"
        NO_DEFAULT_PATH
    )
    if(NOT SLANGC)
        message(FATAL_ERROR "SLANG_ROOT is set but slangc was not found under ${SLANG_ROOT}")
    endif()
endif()

if(NOT SLANGC)
    unset(SLANGC CACHE)
    find_program(SLANGC NAMES slangc slangc.exe)
endif()

if(NOT SLANGC)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(_slang_host_system "windows")
        set(_slang_host_executable_suffix ".exe")
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(_slang_host_system "linux")
        set(_slang_host_executable_suffix "")
    else()
        message(FATAL_ERROR "No Slang compiler download configured for host system ${CMAKE_HOST_SYSTEM_NAME}. Set SLANGC or SLANG_ROOT explicitly.")
    endif()

    set(_slang_host_processor_name "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    if(NOT _slang_host_processor_name AND CMAKE_HOST_WIN32)
        if(DEFINED ENV{PROCESSOR_ARCHITEW6432})
            set(_slang_host_processor_name "$ENV{PROCESSOR_ARCHITEW6432}")
        else()
            set(_slang_host_processor_name "$ENV{PROCESSOR_ARCHITECTURE}")
        endif()
    endif()

    string(TOLOWER "${_slang_host_processor_name}" _slang_host_processor)
    if(_slang_host_processor MATCHES "^(amd64|x86_64)$")
        set(_slang_host_arch "x86_64")
    elseif(_slang_host_processor MATCHES "^(arm64|aarch64)$")
        set(_slang_host_arch "aarch64")
    else()
        message(FATAL_ERROR "Unsupported Slang compiler host architecture: ${_slang_host_processor_name}. Set SLANGC or SLANG_ROOT explicitly.")
    endif()

    if(_slang_host_system STREQUAL "windows" AND _slang_host_arch STREQUAL "x86_64")
        set(_slang_archive_extension "tar.gz")
    else()
        set(_slang_archive_extension "zip")
    endif()

    set(_slang_package_name "slang-${_slang_version}-${_slang_host_system}-${_slang_host_arch}")
    set(_slang_archive_name "${_slang_package_name}.${_slang_archive_extension}")
    string(REGEX REPLACE "/+$" "" _slang_download_base_url "${SLANG_DOWNLOAD_BASE_URL}")
    set(_slang_download_url "${_slang_download_base_url}/v${_slang_version}/${_slang_archive_name}")
    set(_slang_download_dir "${CMAKE_BINARY_DIR}/_deps/slang")
    set(_slang_archive_path "${_slang_download_dir}/${_slang_archive_name}")
    set(_slang_extract_dir "${_slang_download_dir}/${_slang_package_name}")

    file(GLOB_RECURSE _slang_existing_compilers
        "${_slang_extract_dir}/*slangc${_slang_host_executable_suffix}"
    )
    if(NOT _slang_existing_compilers)
        file(MAKE_DIRECTORY "${_slang_download_dir}" "${_slang_extract_dir}")
        if(NOT EXISTS "${_slang_archive_path}")
            message(STATUS "Downloading Slang ${_slang_version} from ${_slang_download_url}")
            file(DOWNLOAD
                "${_slang_download_url}"
                "${_slang_archive_path}"
                STATUS _slang_download_status
                TLS_VERIFY ON
            )
            list(GET _slang_download_status 0 _slang_download_code)
            if(NOT _slang_download_code EQUAL 0)
                list(GET _slang_download_status 1 _slang_download_message)
                message(FATAL_ERROR "Failed to download Slang compiler: ${_slang_download_message}")
            endif()
        endif()

        message(STATUS "Extracting Slang to ${_slang_extract_dir}")
        file(ARCHIVE_EXTRACT
            INPUT "${_slang_archive_path}"
            DESTINATION "${_slang_extract_dir}"
        )
    endif()

    file(GLOB_RECURSE _slang_compiler_candidates
        "${_slang_extract_dir}/*slangc${_slang_host_executable_suffix}"
    )
    list(SORT _slang_compiler_candidates)
    list(LENGTH _slang_compiler_candidates _slang_compiler_count)
    if(_slang_compiler_count EQUAL 0)
        message(FATAL_ERROR "Downloaded Slang archive did not produce slangc in ${_slang_extract_dir}")
    endif()

    list(GET _slang_compiler_candidates 0 SLANGC)
    get_filename_component(_slang_compiler_dir "${SLANGC}" DIRECTORY)
    get_filename_component(_slang_compiler_dir_name "${_slang_compiler_dir}" NAME)
    if(_slang_compiler_dir_name STREQUAL "bin")
        get_filename_component(SLANG_ROOT "${_slang_compiler_dir}" DIRECTORY)
    else()
        set(SLANG_ROOT "${_slang_compiler_dir}")
    endif()

    set(SLANGC "${SLANGC}" CACHE FILEPATH "Path to the Slang compiler executable" FORCE)
    set(SLANG_ROOT "${SLANG_ROOT}" CACHE PATH "Optional path to Slang SDK root directory containing slangc" FORCE)
    message(STATUS "Using downloaded Slang compiler: ${SLANGC}")
else()
    message(STATUS "Using Slang compiler: ${SLANGC}")
endif()
