find_package(Vulkan QUIET)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
include(slang)

option(FLUX2_ENABLE_VULKAN_VALIDATION "Enable Vulkan validation layers for the Flux2 Vulkan target" ON)
option(DIN_BUILD_FLUX2_UI "Build the GLFW/OpenGL Flux2 desktop sample" ON)

add_library(din_flux2_common STATIC
    utils.cpp
)

target_include_directories(din_flux2_common
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/helpers"
)

target_link_libraries(din_flux2_common
    PUBLIC
        din_common_ort
        PRIVATE
        nlohmann_json::nlohmann_json
)

set_target_properties(din_flux2_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

function(configure_flux2_backend_target TARGET)
    set_target_properties(${TARGET} PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
        ARCHIVE_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}"
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
    target_include_directories(${TARGET}
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/helpers"
    )
    target_link_libraries(${TARGET}
        PUBLIC
            din_flux2_common
            din_common_io
    )
endfunction()

function(compile_slang_shader TARGET SHADER_SOURCE OUTPUT_SPV ENTRY_POINT)
    add_custom_command(
        OUTPUT "${OUTPUT_SPV}"
        COMMAND ${SLANGC} "${SHADER_SOURCE}"
                -target spirv -o "${OUTPUT_SPV}"
                -entry "${ENTRY_POINT}" -stage compute
        DEPENDS "${SHADER_SOURCE}"
        COMMENT "g: ${SHADER_SOURCE} -> ${OUTPUT_SPV}"
    )
    get_filename_component(_name "${SHADER_SOURCE}" NAME_WE)
    add_custom_target(${TARGET}_${_name}_spv DEPENDS "${OUTPUT_SPV}")
    add_dependencies(${TARGET} ${TARGET}_${_name}_spv)
endfunction()

add_din_shared_library(din_flux2_cuda
    flux2_cuda_backend.cpp
    cuda_kernels.cu
)
configure_flux2_backend_target(din_flux2_cuda)
set_target_properties(din_flux2_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)
target_link_libraries(din_flux2_cuda PRIVATE CUDA::cudart CUDA::nvtx3)

if(Vulkan_FOUND AND SLANGC)
    add_subdirectory(shaders)

    add_din_shared_library(din_flux2_vk
        flux2_vk_backend.cpp
    )
    configure_flux2_backend_target(din_flux2_vk)
    if(WIN32)
        target_compile_definitions(din_flux2_vk PRIVATE "VK_USE_PLATFORM_WIN32_KHR=1")
    endif()
    target_link_libraries(din_flux2_vk PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::nvtx3
        Vulkan::Vulkan
    )
    target_compile_definitions(din_flux2_vk PRIVATE
        SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders/"
        DIN_ENABLE_VULKAN_VALIDATION=$<BOOL:${FLUX2_ENABLE_VULKAN_VALIDATION}>
    )
    add_dependencies(din_flux2_vk flux2_shaders)
else()
    set(FLUX2_VULKAN_MISSING_DEPS "")
    if(NOT Vulkan_FOUND)
        list(APPEND FLUX2_VULKAN_MISSING_DEPS "Vulkan")
    endif()
    if(NOT SLANGC)
        list(APPEND FLUX2_VULKAN_MISSING_DEPS "Slang compiler (slangc)")
    endif()
    list(JOIN FLUX2_VULKAN_MISSING_DEPS ", " FLUX2_VULKAN_MISSING_DEPS_MESSAGE)
    message(STATUS "Skipping din_flux2_vk; missing dependencies: ${FLUX2_VULKAN_MISSING_DEPS_MESSAGE}")
endif()

if(WIN32)
    add_din_shared_library(din_flux2_dx
        flux2_dx_backend.cpp
    )
    configure_flux2_backend_target(din_flux2_dx)
    target_link_libraries(din_flux2_dx PRIVATE
        d3d12
        dxgi
        d3dcompiler
    )
else()
    message(STATUS "Skipping din_flux2_dx; DirectX 12 target requires Windows")
endif()

add_library(din_flux2_runtime STATIC
    flux2_pipeline.cpp
)
target_include_directories(din_flux2_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(din_flux2_runtime PUBLIC din_flux2_cuda)
if(TARGET din_flux2_dx)
    target_link_libraries(din_flux2_runtime PUBLIC din_flux2_dx)
    target_compile_definitions(din_flux2_runtime PRIVATE DIN_FLUX2_BUILD_DX=1)
endif()
if(TARGET din_flux2_vk)
    target_link_libraries(din_flux2_runtime PUBLIC din_flux2_vk)
    target_compile_definitions(din_flux2_runtime PRIVATE DIN_FLUX2_BUILD_VK=1)
endif()

add_din_executable(din_flux2_cli
    flux2_cli.cpp
)
target_link_libraries(din_flux2_cli PRIVATE argparse din_flux2_runtime)
if (DIN_BUILD_FLUX2_UI)
    include(imgui)
    add_din_ui_executable(din_flux2_ui flux2_ui.cpp)
    target_link_libraries(din_flux2_ui PRIVATE argparse din_flux2_runtime CUDA::cudart)
endif ()
if(TARGET din_flux2_dx)
    target_link_libraries(din_flux2_cli PRIVATE din_flux2_dx)
endif()
if(TARGET din_flux2_vk)
    target_link_libraries(din_flux2_cli PRIVATE din_flux2_vk)
endif()
