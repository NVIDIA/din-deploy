# din::io::Tokenizer unit test. Needs no models, GPU or artifacts.

# Plain add_executable: add_din_executable's post-build DLL copy fails when a target has no runtime DLLs
# (this one only links static libraries).
add_executable(din_tokenizer_test
    test_tokenizer.cpp
)
set_target_properties(din_tokenizer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${DIN_RUNTIME_DIR}")
target_link_libraries(din_tokenizer_test PRIVATE din_common_io nlohmann_json::nlohmann_json)

if(DIN_BUILD_TESTING)
    add_test(NAME din_tokenizer_test COMMAND din_tokenizer_test)
    set_tests_properties(din_tokenizer_test PROPERTIES LABELS "unit")
endif()
