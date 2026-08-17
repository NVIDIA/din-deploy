add_custom_target(flux2_shaders)

set(SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/euler.slang
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.slang
)

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    compile_slang_shader(flux2_shaders ${SHADER_SOURCE} ${SHADER_OUTPUT} main)
endforeach()
