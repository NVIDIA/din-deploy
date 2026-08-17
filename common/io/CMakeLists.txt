include(${PROJECT_SOURCE_DIR}/cmake/lodepng.cmake)

add_library(din_common_io STATIC
    audio.cpp
    image.cpp
    tokenizer.cpp
)

target_include_directories(din_common_io
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(din_common_io
    PUBLIC
        dr_libs::dr_libs
    PRIVATE
        lodepng
        nlohmann_json::nlohmann_json
)

set_target_properties(din_common_io PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

