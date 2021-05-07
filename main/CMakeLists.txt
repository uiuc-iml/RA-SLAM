if (TORCH_FOUND)
    if (CMAKE_CUDA_COMPILER)
        add_executable(live_camera live_camera.cc)
        target_link_libraries(live_camera
            PRIVATE disinf_slam_top_level popl openvslam zed_native l515 ${OpenCV_LIBS})
        target_include_directories(live_camera
            PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        
        # TODO(roger): offline_eval should not strictly depend on ROS.
        if(catkin_FOUND)
            add_executable(offline_eval offline_eval.cc)
            target_link_libraries(offline_eval
            PUBLIC
                gl_utils
                tsdf_module
                renderer_module
                popl
                segmentation
                rotation_math
                openvslam
                scannet_sens_reader
                folder_reader
                rosbag_reader
                ${OpenCV_LIBS})
            target_include_directories(offline_eval
                PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endif ()
    endif ()
endif ()
