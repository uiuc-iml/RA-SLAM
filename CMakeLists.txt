cmake_minimum_required(VERSION 3.8)

project(disinfslam LANGUAGES CXX C)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "-arch=sm_35 -use_fast_math ${CMAKE_CUDA_FLAGS}")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G ${CMAKE_CUDA_FLAGS_DEBUG}")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenGL
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
# Slam
find_package(openvslam REQUIRED)
find_package(Pangolin REQUIRED) # TODO(alvin): to be completely replaced by Dear ImGUI
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
# Zed SDK (optional)
find_package(ZED 3 QUIET)
# Camera SDKs
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

# enable testing
enable_testing()
include(GoogleTest)

# Compile Flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")

include_directories(${CMAKE_SOURCE_DIR})

# Interface wrapper lib for OpenVSLAM
add_library(openvslam INTERFACE)
target_link_libraries(openvslam
    INTERFACE popl openvslam::openvslam openvslam::pangolin_viewer)

add_subdirectory(third_party)
add_subdirectory(utils)
add_subdirectory(cameras)
add_subdirectory(modules)

add_executable(run_sr300 run_sr300.cc)
target_link_libraries(run_sr300 PRIVATE sr300 modules)

add_executable(run_zed_native run_zed_native.cc)
target_link_libraries(run_zed_native PRIVATE zed_native modules)

add_executable(run_zed_native_l515 run_zed_native_l515.cc)
target_link_libraries(run_zed_native_l515 PRIVATE zed_native l515 modules)

if (${ZED_FOUND})
    add_executable(run_zed run_zed.cc)
    target_link_libraries(run_zed PRIVATE zed modules)
endif (${ZED_FOUND})

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(examples)
endif ()
