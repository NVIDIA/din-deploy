# SAM2 video segmentation. Follows the asr/rnnt layering: shared ORT helpers,
# a processing library (din_sam2_pipeline), and a thin CLI.

add_library(din_sam2_pipeline STATIC
    cpp/sam2_io.cpp
    cpp/sam2_pipeline.cpp
)

target_include_directories(din_sam2_pipeline
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)

target_link_libraries(din_sam2_pipeline
    PUBLIC
        lodepng
        nlohmann_json::nlohmann_json
        din_common_ort
        din_common_io
)

set_target_properties(din_sam2_pipeline PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_din_executable(din_sam2_cli
    cpp/sam2_main.cpp
)

target_link_libraries(din_sam2_cli PRIVATE
    din_sam2_pipeline
    argparse
)

if(DIN_BUILD_TESTING)
    add_din_executable(din_sam2_verify_masks
        cpp/sam2_verify_masks.cpp
    )
    target_link_libraries(din_sam2_verify_masks PRIVATE din_sam2_pipeline)

    set(_sam2_test_output "${CMAKE_BINARY_DIR}/sam2_tests")
    set(_sam2_ctest_runner "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_sam2_ctest.cmake")

    add_test(
        NAME din_sam2_image_trt_rtx
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=image
            -DDIN_SAM2_EXE=$<TARGET_FILE:din_sam2_cli>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DEP_CACHE_DIR=${DIN_CLI_TEST_EP_CACHE_DIR}
            -DEP_CONTEXT_DIR=${DIN_CLI_TEST_EP_CONTEXT_DIR}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    add_test(
        NAME din_sam2_image_trt_rtx_verify
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=image_verify
            -DVERIFY_EXE=$<TARGET_FILE:din_sam2_verify_masks>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    set_tests_properties(din_sam2_image_trt_rtx_verify PROPERTIES DEPENDS din_sam2_image_trt_rtx)

    add_test(
        NAME din_sam2_video_trt_rtx
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=video
            -DDIN_SAM2_EXE=$<TARGET_FILE:din_sam2_cli>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DEP_CACHE_DIR=${DIN_CLI_TEST_EP_CACHE_DIR}
            -DEP_CONTEXT_DIR=${DIN_CLI_TEST_EP_CONTEXT_DIR}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    add_test(
        NAME din_sam2_video_trt_rtx_verify
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_KIND=video_verify
            -DVERIFY_EXE=$<TARGET_FILE:din_sam2_verify_masks>
            -DDEFAULT_ARTIFACTS_ROOT=${DIN_DEPLOY_ARTIFACTS_ROOT}
            -DOUTPUT_DIR=${_sam2_test_output}
            -P "${_sam2_ctest_runner}"
    )
    set_tests_properties(din_sam2_video_trt_rtx_verify PROPERTIES DEPENDS din_sam2_video_trt_rtx)

    set_tests_properties(
        din_sam2_image_trt_rtx
        din_sam2_image_trt_rtx_verify
        din_sam2_video_trt_rtx
        din_sam2_video_trt_rtx_verify
        PROPERTIES LABELS "sam2;cli"
    )

    unset(_sam2_ctest_runner)
    unset(_sam2_test_output)
endif()
