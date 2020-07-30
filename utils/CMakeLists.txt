add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PRIVATE spdlog opencv_imgproc opencv_calib3d openvslam::openvslam)

add_subdirectory(gl)
add_subdirectory(tsdf)

add_subdirectory(tests)
