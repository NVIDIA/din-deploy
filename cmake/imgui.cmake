include_guard(GLOBAL)
include(FetchContent)

# Attach the shared GLFW/OpenGL ImGui implementation to an existing UI target.
# Dependencies are fetched only when a UI target is requested.
function(din_link_imgui TARGET)
    if (NOT TARGET din_imgui)
        find_package(OpenGL REQUIRED GLOBAL)
        find_package(Threads REQUIRED GLOBAL)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        if (UNIX AND NOT APPLE)
            set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
            set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
        endif ()
        FetchContent_Declare(glfw
                GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
        FetchContent_Declare(imgui
                GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.5 GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(glfw imgui)
        add_library(din_imgui STATIC
                ${imgui_SOURCE_DIR}/imgui.cpp
                ${imgui_SOURCE_DIR}/imgui_draw.cpp
                ${imgui_SOURCE_DIR}/imgui_tables.cpp
                ${imgui_SOURCE_DIR}/imgui_widgets.cpp
                ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
                ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
        target_include_directories(din_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
        target_link_libraries(din_imgui PUBLIC glfw OpenGL::GL Threads::Threads)
    endif ()
    target_link_libraries(${TARGET} PRIVATE din_imgui)
endfunction()

# Usage: add_din_ui_executable(my_app main.cpp [other sources...])
# App-specific libraries and settings belong in the caller's CMakeLists.txt.
function(add_din_ui_executable TARGET)
    add_din_executable(${TARGET} ${ARGN})
    din_link_imgui(${TARGET})
endfunction()
