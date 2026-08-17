# Whisper speech-to-text. Follows the asr/rnnt layering: a processing library
# (din_asr_whisper) plus a thin CLI (din_asr_whisper)

# Optional CUDA greedy-argmax kernel for the TRT-RTX decode path. When CUDA is
# unavailable the pipeline falls back to a pinned-host CPU argmax (DIN_WHISPER_CUDA
# undefined), so this is purely additive.
find_package(CUDAToolkit QUIET)
set(_whisper_cuda_sources)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    list(APPEND _whisper_cuda_sources whisper_kernels.cu)
endif()

add_din_shared_library(din_asr_whisper STATIC
    whisper.cpp
    ${_whisper_cuda_sources}
)

target_include_directories(din_asr_whisper
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(din_asr_whisper
    PUBLIC
        din_common_ort
        din_common_io
        nlohmann_json::nlohmann_json
)

set_target_properties(din_asr_whisper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(CUDAToolkit_FOUND)
    target_link_libraries(din_asr_whisper PRIVATE CUDA::cudart)
    target_compile_definitions(din_asr_whisper PRIVATE DIN_WHISPER_CUDA=1)
    set_target_properties(din_asr_whisper PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
endif()

add_din_executable(din_asr_whisper_cli
    main.cpp
)

target_link_libraries(din_asr_whisper_cli
    PRIVATE
        din_asr_whisper
        argparse
)
