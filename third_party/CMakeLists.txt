add_library(popl INTERFACE)
target_include_directories(popl SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/popl/include)

add_subdirectory(scannet)
add_subdirectory(imgui-1.77)

add_subdirectory(googletest-1.10.0)