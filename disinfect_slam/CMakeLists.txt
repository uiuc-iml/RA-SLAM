add_library(disinf_slam_top_level disinfect_slam.cc)
target_link_libraries(disinf_slam_top_level
    PUBLIC gl_utils renderer_module tsdf_module slam_module popl zed_native l515 segmentation rotation_math ${OpenCV_LIBS})
target_include_directories(disinf_slam_top_level PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})