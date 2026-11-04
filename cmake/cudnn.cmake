# cuDNN backend and header-only cuDNN frontend targets.
set(_cuDNN_DEFAULT_ROOT "")
if(DEFINED ENV{cuDNN_ROOT})
    set(_cuDNN_DEFAULT_ROOT "$ENV{cuDNN_ROOT}")
elseif(DEFINED ENV{CUDNN_ROOT})
    set(_cuDNN_DEFAULT_ROOT "$ENV{CUDNN_ROOT}")
elseif(CUDNN_ROOT)
    set(_cuDNN_DEFAULT_ROOT "${CUDNN_ROOT}")
endif()
set(cuDNN_ROOT "${_cuDNN_DEFAULT_ROOT}" CACHE PATH "cuDNN SDK installation root")

find_package(CUDAToolkit QUIET)
find_path(cuDNN_INCLUDE_DIR cudnn.h HINTS "${cuDNN_ROOT}" PATH_SUFFIXES include)
find_library(cuDNN_LIBRARY NAMES cudnn cudnn64_9
    HINTS "${cuDNN_ROOT}" PATH_SUFFIXES lib lib/x64 lib64)

if(EXISTS "${cuDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ "${cuDNN_INCLUDE_DIR}/cudnn_version.h" _cuDNN_HEADER)
    foreach(part MAJOR MINOR PATCHLEVEL)
        string(REGEX MATCH "#define[ \t]+CUDNN_${part}[ \t]+([0-9]+)"
            _match "${_cuDNN_HEADER}")
        set(_cuDNN_${part} "${CMAKE_MATCH_1}")
    endforeach()
    set(cuDNN_VERSION "${_cuDNN_MAJOR}.${_cuDNN_MINOR}.${_cuDNN_PATCHLEVEL}")
endif()

set(_cuDNN_REQUIRED_VARS cuDNN_INCLUDE_DIR cuDNN_LIBRARY CUDAToolkit_FOUND)
if(WIN32)
    get_filename_component(_cuDNN_LIBRARY_DIR "${cuDNN_LIBRARY}" DIRECTORY)
    find_file(cuDNN_RUNTIME_LIBRARY "cudnn64_${_cuDNN_MAJOR}.dll"
        HINTS "${cuDNN_ROOT}" "${_cuDNN_LIBRARY_DIR}/.." "${_cuDNN_LIBRARY_DIR}/../.."
        PATH_SUFFIXES bin bin/x64 lib lib/x64)
    list(APPEND _cuDNN_REQUIRED_VARS cuDNN_RUNTIME_LIBRARY)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
    REQUIRED_VARS ${_cuDNN_REQUIRED_VARS}
    VERSION_VAR cuDNN_VERSION
    REASON_FAILURE_MESSAGE "Set cuDNN_ROOT to the cuDNN SDK installation root and install CUDA Toolkit.")

if(cuDNN_FOUND AND NOT TARGET cuDNN::cuDNN)
    add_library(cuDNN::cuDNN SHARED IMPORTED GLOBAL)
    set_target_properties(cuDNN::cuDNN PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES CUDA::cudart)
    if(WIN32)
        set_target_properties(cuDNN::cuDNN PROPERTIES
            IMPORTED_IMPLIB "${cuDNN_LIBRARY}"
            IMPORTED_LOCATION "${cuDNN_RUNTIME_LIBRARY}")
    else()
        set_target_properties(cuDNN::cuDNN PROPERTIES IMPORTED_LOCATION "${cuDNN_LIBRARY}")
    endif()
endif()

option(cuDNN_FETCH_FRONTEND "Fetch cuDNN frontend and expose cuDNN::frontend" ON)
if(cuDNN_FOUND AND cuDNN_FETCH_FRONTEND AND NOT TARGET cuDNN::frontend)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    # Populate headers only: avoid upstream's samples, tests, install rules, and
    # compiler flags. SOURCE_SUBDIR deliberately has no CMakeLists.txt.
    FetchContent_Declare(cudnn_frontend
        URL https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/refs/tags/v1.32.0
        URL_HASH SHA256=f4fe84095f848dd77b148feb592bc4ac4a3b085cd3fe1d9e95e1911e846f7ed1
        SOURCE_SUBDIR _din_headers_only
        TIMEOUT 60
        INACTIVITY_TIMEOUT 30)
    FetchContent_MakeAvailable(cudnn_frontend)
    add_library(cuDNN::frontend INTERFACE IMPORTED GLOBAL)
    set_target_properties(cuDNN::frontend PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${cudnn_frontend_SOURCE_DIR}/include"
        INTERFACE_COMPILE_FEATURES cxx_std_17
        INTERFACE_LINK_LIBRARIES "cuDNN::cuDNN;CUDA::cudart;${CMAKE_DL_LIBS}")
endif()

mark_as_advanced(cuDNN_INCLUDE_DIR cuDNN_LIBRARY cuDNN_RUNTIME_LIBRARY)
