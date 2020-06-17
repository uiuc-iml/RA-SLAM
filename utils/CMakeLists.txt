add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PRIVATE opencv_imgproc opencv_calib3d openvslam::openvslam)
