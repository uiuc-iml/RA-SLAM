add_library(modules slam_module.cc tsdf_module.cc)
target_link_libraries(modules PUBLIC openvslam)

