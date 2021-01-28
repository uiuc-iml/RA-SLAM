add_library(slam_module slam_module.cc)
target_link_libraries(slam_module PUBLIC openvslam)

add_library(renderer_module renderer_module.cc)
target_link_libraries(renderer_module PRIVATE gl_utils tsdf_module slam_module popl)
target_include_directories(renderer_module PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(tsdf_module tsdf_module.cc)
target_link_libraries(tsdf_module PUBLIC tsdf_utils gl_utils
                                  PRIVATE openvslam)
