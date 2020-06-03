cmake_minimum_required(VERSION 3.8)
project(disinfslam LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(openvslam REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)

add_subdirectory(third_party)

add_executable(run_realsense2 run_realsense2.cc sr300.cc)
target_link_libraries(run_realsense2 PRIVATE popl
    openvslam::openvslam
    openvslam::pangolin_viewer
    ${realsense2_LIBRARY})
target_include_directories(run_realsense2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
