add_library(scannet_sens_reader scannet_sens_reader.cc)
target_link_libraries(scannet_sens_reader PUBLIC cuda_util ${OPENCV_LIBS} scannet)

add_library(folder_reader folder_reader.cc)
target_link_libraries(folder_reader PUBLIC cuda_util ${OPENCV_LIBS} openvslam)

if(catkin_FOUND)
    add_library(rosbag_reader rosbag_reader.cc)
    target_link_libraries(rosbag_reader
        PUBLIC
        cuda_util
        ${OPENCV_LIBS}
        ${catkin_LIBRARIES}
        openvslam 
        rotation_math
        slam_module
        utils
    )
endif()
