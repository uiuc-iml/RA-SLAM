add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PUBLIC opencv_imgproc opencv_calib3d yaml-cpp
                            PRIVATE openvslam::openvslam)

add_subdirectory(cuda)
add_subdirectory(gl)
add_subdirectory(tsdf)
add_subdirectory(tests)

add_subdirectory(rotation_math)
