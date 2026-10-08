find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# The sampling math mirrors the PyTorch reference operation by operation in fp32, on the host (--processing cpu)
# and in cuda_kernels.cu (--processing cuda); keep both compilers from fusing multiply-adds so the two paths give
# identical results (MSVC /fp:precise does not contract by default).
set(PIXELDIT_FP_FLAGS
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
)

add_library(din_pixeldit_common STATIC
    pixeldit_config.cpp
)

target_include_directories(din_pixeldit_common
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/helpers"
)

target_link_libraries(din_pixeldit_common
    PUBLIC
        din_common_ort
        din_common_io
    PRIVATE
        nlohmann_json::nlohmann_json
)

set_target_properties(din_pixeldit_common PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

add_din_shared_library(din_pixeldit_trt
    pixeldit_trt_backend.cpp
    cuda_kernels.cu
)
set_target_properties(din_pixeldit_trt PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)
target_compile_options(din_pixeldit_trt PRIVATE ${PIXELDIT_FP_FLAGS})
target_link_libraries(din_pixeldit_trt
    PUBLIC
        din_pixeldit_common
    PRIVATE
        CUDA::cudart
)

add_library(din_pixeldit_runtime STATIC
    pixeldit_pipeline.cpp
)
set_target_properties(din_pixeldit_runtime PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)
target_link_libraries(din_pixeldit_runtime PUBLIC din_pixeldit_trt)

add_din_executable(din_pixeldit_cli
    pixeldit_cli.cpp
)
set_target_properties(din_pixeldit_cli PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)
target_link_libraries(din_pixeldit_cli PRIVATE argparse din_pixeldit_runtime)
