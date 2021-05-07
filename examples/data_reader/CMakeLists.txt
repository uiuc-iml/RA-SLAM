add_executable(sens_reader_manual_test sens_reader_manual_test.cc)
target_link_libraries(sens_reader_manual_test cuda_util ${OpenCV_LIBS} segmentation scannet_sens_reader)

if(catkin_FOUND)
    add_executable(rosbag_reader_manual_test rosbag_reader_manual_test.cc)
    target_link_libraries(rosbag_reader_manual_test cuda_util ${OpenCV_LIBS} segmentation rosbag_reader)
endif()