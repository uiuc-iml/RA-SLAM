cmake_minimum_required(VERSION 3.8)
project(disinfslam LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(openvslam REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)

# Interface wrapper lib for OpenVSLAM
add_library(openvslam INTERFACE)
target_link_libraries(openvslam
    INTERFACE popl openvslam::openvslam openvslam::pangolin_viewer)

# options
option(WITH_ZED_SDK "Whether to include ZED SDK or not (disable on MacOS)" ON)

add_subdirectory(cameras)
add_subdirectory(modules)
add_subdirectory(third_party)

add_executable(run_sr300 run_sr300.cc)
target_link_libraries(run_sr300 PRIVATE sr300 openvslam)

add_executable(run_zed run_zed.cc)
target_link_libraries(run_zed PRIVATE zed_native openvslam modules)
