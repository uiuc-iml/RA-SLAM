# From parent CMake
# It's safe to assume that libtorch is installed.
if (CMAKE_CUDA_COMPILER)
    add_executable(interfaced_recon_example main.cc)
    target_link_libraries(interfaced_recon_example
        PRIVATE disinf_slam_top_level popl openvslam zed_native l515 ${OpenCV_LIBS})
    add_dependencies(interfaced_recon_example disinf_slam_top_level)
    target_include_directories(interfaced_recon_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()
