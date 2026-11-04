find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

option(DIN_SEEDVR_NATIVE_VAE "Build the native cuDNN SeedVR2 VAE" ON)
if(DIN_SEEDVR_NATIVE_VAE)
    include(cudnn)
    add_library(din_seedvr2_vae STATIC seedvr_vae.cpp seedvr_vae_kernels.cu)
    target_include_directories(din_seedvr2_vae PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" PRIVATE "${PROJECT_SOURCE_DIR}/common")
    if(DIN_ENABLE_NVTX)
        target_link_libraries(din_seedvr2_vae PRIVATE nvtx3-cpp)
        target_compile_definitions(din_seedvr2_vae PRIVATE DIN_ENABLE_NVTX=1)
    endif()
    target_link_libraries(din_seedvr2_vae PUBLIC CUDA::cudart PRIVATE CUDA::cuda_driver CUDA::nvrtc cuDNN::frontend nlohmann_json::nlohmann_json)
    set_target_properties(din_seedvr2_vae PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}" POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(din_seedvr2_vae PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    add_din_executable(din_seedvr2_vae_cli seedvr_vae_cli.cpp)
    target_link_libraries(din_seedvr2_vae_cli PRIVATE din_seedvr2_vae)
    if(WIN32)
        get_filename_component(_seedvr_cudnn_bin "${cuDNN_RUNTIME_LIBRARY}" DIRECTORY)
        file(GLOB _seedvr_cudnn_dlls "${_seedvr_cudnn_bin}/cudnn*.dll")
        add_custom_command(TARGET din_seedvr2_vae_cli POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_seedvr_cudnn_dlls} $<TARGET_FILE_DIR:din_seedvr2_vae_cli> COMMAND_EXPAND_LISTS)
    endif()
endif()

add_library(din_seedvr2_pipeline STATIC
        seedvr_pipeline.cpp
        seedvr_cuda_kernels.cu
)

target_include_directories(din_seedvr2_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(DIN_SEEDVR_NATIVE_VAE)
    target_link_libraries(din_seedvr2_pipeline PRIVATE din_seedvr2_vae)
    target_compile_definitions(din_seedvr2_pipeline PRIVATE DIN_SEEDVR_NATIVE_VAE=1)
endif()
target_link_libraries(din_seedvr2_pipeline
        PUBLIC
        din_common_ort
        din_common_rng
        din_common_rng_cuda
        PRIVATE
        CUDA::cudart
)
set_target_properties(din_seedvr2_pipeline PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)
if(MSVC)
    target_compile_options(din_seedvr2_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

add_din_executable(din_seedvr2_cli seedvr_cli.cpp)
target_link_libraries(din_seedvr2_cli PRIVATE argparse din_seedvr2_pipeline din_common_io lodepng)
if(DIN_SEEDVR_NATIVE_VAE AND WIN32)
    add_custom_command(TARGET din_seedvr2_cli POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_seedvr_cudnn_dlls} $<TARGET_FILE_DIR:din_seedvr2_cli> COMMAND_EXPAND_LISTS)
endif()
