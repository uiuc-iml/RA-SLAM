if (CMAKE_CUDA_COMPILER)
    add_executable(simple_renderer main.cc)
    target_link_libraries(simple_renderer gl_utils)
    get_filename_component(VERTEX_SHADER_PATH simple_renderer/shader.vert ABSOLUTE)
    get_filename_component(FRAGMENT_SHADER_PATH simple_renderer/shader.frag ABSOLUTE)
    target_compile_definitions(simple_renderer PRIVATE
        VERTEX_SHADER_PATH="${VERTEX_SHADER_PATH}" FRAGMENT_SHADER_PATH="${FRAGMENT_SHADER_PATH}")
endif ()
