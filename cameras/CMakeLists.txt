find_package(realsense2 REQUIRED)
add_library(sr300 sr300.cc)
target_link_libraries(sr300
    PUBLIC realsense2::realsense2
    PRIVATE openvslam::openvslam)

find_package(OpenCV REQUIRED)
add_library(zed_native zed_native.cc)
target_link_libraries(zed_native PUBLIC ${OpenCV_LIBS} openvslam::openvslam utils)

add_library(zed_sdk INTERFACE)
target_link_libraries(zed_sdk INTERFACE ${ZED_LIBRARIES} ${CUDA_LIBRARY})
target_include_directories(zed_sdk INTERFACE ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

if (${ZED_FOUND})
    add_library(zed zed.cc)
    target_link_libraries(zed
        PUBLIC zed_sdk
        PRIVATE openvslam::openvslam)
endif (${ZED_FOUND})
