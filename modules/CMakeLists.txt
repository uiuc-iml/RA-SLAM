add_library(slam_module slam_module.cc)
target_link_libraries(slam_module PUBLIC openvslam)

add_library(tsdf_module tsdf_module.cc)
target_link_libraries(tsdf_module PUBLIC tsdf_utils gl_utils
                                  PRIVATE openvslam)

