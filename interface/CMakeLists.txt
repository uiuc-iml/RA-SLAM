cmake_minimum_required(VERSION 2.8)
project(disinfect-slam-dummy-interface)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INLUCDE_DIRS})

add_library(disinfect_slam_top_level top_level.cc)

add_executable(dummy_main ./main.cc)

target_link_libraries(dummy_main ${OpenCV_LIBS} disinfect_slam_top_level)