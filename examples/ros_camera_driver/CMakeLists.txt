if (CMAKE_CUDA_COMPILER)
    add_executable(ros_disinf_slam main.cc ros_interface.cc ros_test.cc)
    target_link_libraries(ros_disinf_slam
        PUBLIC ${catkin_LIBRARIES} ${KLAMPT_LIBRARIES} -L/home/motion/.local/redis-plus-plus/lib -lredis++ -L/home/motion/.local/hiredis/lib -lhiredis -pthread
        PRIVATE disinf_slam_top_level popl openvslam zed_native l515 ${OpenCV_LIBS}
    )
    add_dependencies(ros_disinf_slam disinf_slam_top_level)
    target_include_directories(ros_disinf_slam PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()