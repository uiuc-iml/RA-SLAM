add_library(gl_utils renderer_base.cc shader.cc image.cc)
target_link_libraries(gl_utils 
    PUBLIC imgui
    PRIVATE spdlog)
