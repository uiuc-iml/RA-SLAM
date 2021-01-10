add_library(popl INTERFACE)
target_include_directories(popl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/popl/include)

add_subdirectory(imgui-1.77)
#add_subdirectory(spdlog-1.6.1)
add_subdirectory(googletest-1.10.0)
