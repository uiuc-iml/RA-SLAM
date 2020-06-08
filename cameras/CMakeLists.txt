find_package(realsense2 REQUIRED)
add_library(sr300 sr300.cc)
target_link_libraries(sr300
    PUBLIC realsense2::realsense2
    PRIVATE openvslam::openvslam)

find_package(ZED 3 REQUIRED)
find_package(CUDA REQUIRED)
add_library(zed_sdk INTERFACE)
target_link_libraries(zed_sdk INTERFACE ${ZED_LIBRARIES} ${CUDA_LIBRARY})
target_include_directories(zed_sdk INTERFACE ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_library(zed zed.cc)
target_link_libraries(zed
    PUBLIC zed_sdk
    PRIVATE openvslam::openvslam)
