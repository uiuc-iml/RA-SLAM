# simple_renderer
add_executable(simple_renderer simple_renderer/main.cc)
target_link_libraries(simple_renderer gl_utils)
get_filename_component(VERTEX_SHADER_PATH simple_renderer/shader.vert ABSOLUTE)
get_filename_component(FRAGMENT_SHADER_PATH simple_renderer/shader.frag ABSOLUTE)
target_compile_definitions(simple_renderer PRIVATE
    VERTEX_SHADER_PATH="${VERTEX_SHADER_PATH}" FRAGMENT_SHADER_PATH="${FRAGMENT_SHADER_PATH}")

add_executable(tsdf_sephere tsdf_sephere/main.cu)
target_link_libraries(tsdf_sephere PRIVATE tsdf_utils ${OpenCV_LIBS})
set_target_properties(tsdf_sephere PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(tsdf_offline tsdf_offline/main.cc)
target_link_libraries(tsdf_offline PRIVATE tsdf_utils popl spdlog zed ${OpenCV_LIBS})
target_include_directories(tsdf_offline PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
