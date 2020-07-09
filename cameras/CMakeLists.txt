add_library(sr300 sr300.cc)
target_link_libraries(sr300
    PUBLIC realsense2::realsense2 ${OpenCV_LIBS}
    PRIVATE spdlog)

add_library(zed_native zed_native.cc)
target_link_libraries(zed_native PUBLIC ${OpenCV_LIBS} openvslam::openvslam utils)

add_library(zed_sdk INTERFACE)
target_link_libraries(zed_sdk INTERFACE ${ZED_LIBRARIES})
target_include_directories(zed_sdk INTERFACE 
    ${ZED_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(zed_sdk INTERFACE -Wno-deprecated-declarations)

if (${ZED_FOUND})
    add_library(zed zed.cc)
    target_link_libraries(zed
        PUBLIC zed_sdk
        PRIVATE openvslam::openvslam)
endif (${ZED_FOUND})
