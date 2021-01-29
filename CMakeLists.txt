cmake_minimum_required(VERSION 3.18)

project(disinfslam LANGUAGES CXX C)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "-use_fast_math ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS_DEBUG "-G ${CMAKE_CUDA_FLAGS_DEBUG}")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 61 72 75 80)

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
    add_subdirectory(disinfect_slam)
endif ()
add_subdirectory(examples)

# -------------------------------------------------------------------- #
# ----------------------- clang format target ------------------------ #
# -------------------------------------------------------------------- #

find_program(CLANG_FORMAT_EXE NAMES
    clang-format-10
    clang-format-9
    clang-format-8
    clang-format)

# gather all source clode
file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/*.cc
    ${CMAKE_SOURCE_DIR}/*.h
    ${CMAKE_SOURCE_DIR}/*.hpp
    ${CMAKE_SOURCE_DIR}/*.cu
    ${CMAKE_SOURCE_DIR}/*.cuh
)

# exclude some directories
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX .*/.*build.*/.*)
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX .*/third_party/.*)

# create formatting helper targets
if (CLANG_FORMAT_EXE) 
    set(RUN_CLANG_FORMAT ${CMAKE_SOURCE_DIR}/run-clang-format.py)
    # format code in place
    add_custom_target(format
        COMMAND python3 ${RUN_CLANG_FORMAT} --clang-format-executable ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
        DEPENDS ${RUN_CLANG_FORMAT} 
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    # check for format violations
    add_custom_target(check-format
        COMMAND python3 ${RUN_CLANG_FORMAT} --clang-format-executable ${CLANG_FORMAT_EXE} ${ALL_SOURCE_FILES}
        DEPENDS ${RUN_CLANG_FORMAT} 
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif (CLANG_FORMAT_EXE)
