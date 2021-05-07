add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PUBLIC opencv_imgproc opencv_calib3d yaml-cpp
                            PRIVATE openvslam::openvslam)

add_subdirectory(cuda)
add_subdirectory(gl)
add_subdirectory(tsdf)

# TODO(roger): figure out why tests can't be compiled in ROS
if(NOT catkin_FOUND)
    add_subdirectory(tests)
endif()

add_subdirectory(rotation_math)
add_subdirectory(offline_data_provider)
