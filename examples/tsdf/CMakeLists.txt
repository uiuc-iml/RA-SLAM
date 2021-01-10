if (CMAKE_CUDA_COMPILER)
    add_executable(offline offline.cc)
    target_link_libraries(offline PRIVATE gl_utils tsdf_utils popl openvslam zed l515 ${OpenCV_LIBS})
    target_include_directories(offline PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_executable(online online.cc)
    target_link_libraries(online
        PRIVATE gl_utils tsdf_module slam_module popl zed_native l515 ${OpenCV_LIBS})
    target_include_directories(online PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()
