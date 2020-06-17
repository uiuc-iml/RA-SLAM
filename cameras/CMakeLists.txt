find_package(realsense2 REQUIRED)
add_library(sr300 sr300.cc)
target_link_libraries(sr300
    PUBLIC realsense2::realsense2
    PRIVATE openvslam::openvslam)

find_package(OpenCV REQUIRED)
add_library(zed_native zed_native.cc)
target_link_libraries(zed_native PUBLIC ${OpenCV_LIBS} openvslam::openvslam utils)
