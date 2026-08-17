
# =============================================================================
# Model download
# =============================================================================
set(RESNET18_MODEL_NAME "resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_URL
        "https://huggingface.co/onnxmodelzoo/resnet18_Opset18_timm/resolve/main/resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RESNET18_MODEL_NAME}")

if(NOT EXISTS "${RESNET18_MODEL_PATH}")
    message(STATUS "Downloading ${RESNET18_MODEL_NAME} ...")
    file(DOWNLOAD "${RESNET18_MODEL_URL}" "${RESNET18_MODEL_PATH}"
            STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(WARNING "Failed to download ${RESNET18_MODEL_NAME} (${_dl_status}). "
                "test_load_model will fail at runtime.")
        file(REMOVE "${RESNET18_MODEL_PATH}")
    endif()
endif()



add_din_executable(din_base_onnx
    main.cpp
)

target_compile_definitions(din_base_onnx PRIVATE RESNET18_MODEL_PATH=\"${RESNET18_MODEL_PATH}\")
target_link_libraries(din_base_onnx PRIVATE argparse din_common_ort)
