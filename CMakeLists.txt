cmake_minimum_required(VERSION 3.11)

project(disinfslam LANGUAGES CXX C)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "-arch=sm_35 -use_fast_math ${CMAKE_CUDA_FLAGS}")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G ${CMAKE_CUDA_FLAGS_DEBUG}")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
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
# DL Inference
find_package(Torch QUIET HINTS ${CMAKE_SOURCE_DIR}/third_party/libtorch)
if (NOT Torch_FOUND)
    MESSAGE(WARNING "libtorch not found. Excluding DL inference engine and examples.")
endif()

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
if (TORCH_FOUND)
    add_subdirectory(segmentation)
endif ()
add_subdirectory(examples)

