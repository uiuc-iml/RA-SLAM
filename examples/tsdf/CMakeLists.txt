add_executable(offline offline.cc)
target_link_libraries(offline PRIVATE gl_utils tsdf_utils popl openvslam zed l515 ${OpenCV_LIBS})
target_include_directories(offline PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if (TORCH_FOUND)
    add_executable(online online.cc)
    target_link_libraries(online
        PRIVATE tsdf_module renderer_module slam_module popl zed_native l515 segmentation rotation_math ${OpenCV_LIBS})
    target_include_directories(online PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()

add_executable(online_no_seg online_no_seg.cc)
target_link_libraries(online_no_seg
    PRIVATE tsdf_module renderer_module slam_module popl zed_native l515 rotation_math ${OpenCV_LIBS})
target_include_directories(online_no_seg PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
