set(SOURCES imgui.cpp imgui_demo.cpp imgui_draw.cpp imgui_widgets.cpp
            imgui_impl_glfw.cpp imgui_impl_opengl3.cpp)

add_library(imgui ${SOURCES})
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(imgui PUBLIC GLEW::GLEW glfw OpenGL::GL)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Official Examples for glfw3 + glew + OpenGL3
# add_executable(imgui_example_glfw_opengl3 example_glfw_opengl3/main.cpp)
# target_link_libraries(imgui_example_glfw_opengl3 imgui)
