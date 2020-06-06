project(cameras)

add_library(${PROJECT_NAME} sr300.cc)
target_link_libraries(${PROJECT_NAME}
    PRIVATE realsense2::realsense2 openvslam::openvslam)
