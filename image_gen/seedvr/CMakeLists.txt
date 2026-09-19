find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_library(din_seedvr2_pipeline STATIC
        seedvr_pipeline.cpp
        seedvr_cuda_kernels.cu
)

target_include_directories(din_seedvr2_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(din_seedvr2_pipeline
        PUBLIC
        din_common_ort
        din_common_io
        PRIVATE
        CUDA::cudart
        lodepng
)
set_target_properties(din_seedvr2_pipeline PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)

add_din_executable(din_seedvr2_cli seedvr_cli.cpp)
target_link_libraries(din_seedvr2_cli PRIVATE argparse din_seedvr2_pipeline)
