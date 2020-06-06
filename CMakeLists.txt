cmake_minimum_required(VERSION 3.8)
project(disinfslam LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(openvslam REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(third_party)
add_subdirectory(cameras)

add_executable(run_sr300 run_sr300.cc)
target_link_libraries(run_sr300 PRIVATE cameras
    popl
    openvslam::openvslam
    openvslam::pangolin_viewer
    realsense2::realsense2)
