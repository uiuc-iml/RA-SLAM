add_library(popl INTERFACE)
target_include_directories(popl SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/popl/include)

add_subdirectory(scannet)
add_subdirectory(imgui-1.77)
#add_subdirectory(spdlog-1.6.1)

# ROS comes with gtest. This causes conflict with ROS.
if (NOT catkin_FOUND)
    add_subdirectory(googletest-1.10.0)
endif ()

add_subdirectory(lcm-1.4.0)