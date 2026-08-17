option(DIN_ENABLE_PARAKEET_PYTHON_BINDINGS "Build Parakeet TDT nanobind Python bindings" OFF)
option(DIN_ENABLE_NEMOTRON_PYTHON_BINDINGS "Build Nemotron ASR nanobind Python bindings" OFF)

add_din_shared_library(din_asr_parakeet_tdt SHARED
    cpp/parakeet_tdt.cpp
)

target_include_directories(din_asr_parakeet_tdt
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)
target_link_libraries(din_asr_parakeet_tdt
    PUBLIC
        din_common_ort
        din_common_io
        nlohmann_json::nlohmann_json
        argparse
)

set_target_properties(din_asr_parakeet_tdt PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_din_executable(din_asr_parakeet_tdt_cli
    cpp/parakeet_tdt_main.cpp
)
target_link_libraries(din_asr_parakeet_tdt_cli PRIVATE din_asr_parakeet_tdt)

add_din_shared_library(din_asr_nemotron SHARED
    cpp/nemotron.cpp
)

target_include_directories(din_asr_nemotron
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)
target_link_libraries(din_asr_nemotron
    PUBLIC
        din_common_ort
        din_common_io
        nlohmann_json::nlohmann_json
        argparse
)

set_target_properties(din_asr_nemotron PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_din_executable(din_asr_nemotron_cli
    cpp/nemotron_main.cpp
)
target_link_libraries(din_asr_nemotron_cli PRIVATE din_asr_nemotron)

if(DIN_ENABLE_PARAKEET_PYTHON_BINDINGS OR DIN_ENABLE_NEMOTRON_PYTHON_BINDINGS)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.9.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()

function(copy_rnnt_python_runtime target package_name)
    set(_rnnt_python_runtime_files)
    if(WIN32)
        if(ONNXRUNTIME_DLL)
            list(APPEND _rnnt_python_runtime_files "${ONNXRUNTIME_DLL}")
        endif()
        if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
            list(APPEND _rnnt_python_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}")
        endif()
    else()
        if(ONNXRUNTIME_LIB)
            list(APPEND _rnnt_python_runtime_files "${ONNXRUNTIME_LIB}")
            get_filename_component(_onnxruntime_lib_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
            file(GLOB _onnxruntime_soname_files "${_onnxruntime_lib_dir}/libonnxruntime.so.*")
            list(APPEND _rnnt_python_runtime_files ${_onnxruntime_soname_files})
        endif()
        if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
            list(APPEND _rnnt_python_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_LIB}")
        endif()
    endif()

    if(_rnnt_python_runtime_files)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    ${_rnnt_python_runtime_files}
                    "$<TARGET_FILE_DIR:${target}>"
            COMMAND_EXPAND_LISTS
        )
        install(FILES ${_rnnt_python_runtime_files} DESTINATION ${package_name})
    endif()
endfunction()

if(DIN_ENABLE_PARAKEET_PYTHON_BINDINGS)
    nanobind_add_module(_parakeet_tdt_cpp
        STABLE_ABI
        NB_STATIC
        cpp/parakeet_tdt_python_bindings.cpp
    )
    target_link_libraries(_parakeet_tdt_cpp PRIVATE din_asr_parakeet_tdt)
    set_target_properties(_parakeet_tdt_cpp PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
    install(TARGETS _parakeet_tdt_cpp LIBRARY DESTINATION rnnt RUNTIME DESTINATION rnnt)
    install(TARGETS din_asr_parakeet_tdt LIBRARY DESTINATION rnnt RUNTIME DESTINATION rnnt)
    copy_rnnt_python_runtime(_parakeet_tdt_cpp rnnt)
endif()

if(DIN_ENABLE_NEMOTRON_PYTHON_BINDINGS)
    nanobind_add_module(_nemotron_asr_cpp
        STABLE_ABI
        NB_STATIC
        cpp/nemotron_python_bindings.cpp
    )
    target_link_libraries(_nemotron_asr_cpp PRIVATE din_asr_nemotron)
    set_target_properties(_nemotron_asr_cpp PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
    install(TARGETS _nemotron_asr_cpp LIBRARY DESTINATION rnnt RUNTIME DESTINATION rnnt)
    install(TARGETS din_asr_nemotron LIBRARY DESTINATION rnnt RUNTIME DESTINATION rnnt)
    copy_rnnt_python_runtime(_nemotron_asr_cpp rnnt)
endif()
