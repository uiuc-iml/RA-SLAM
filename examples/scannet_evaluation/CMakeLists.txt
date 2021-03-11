add_executable(eval_one eval_one.cc)
target_link_libraries(eval_one
    PRIVATE gl_utils tsdf_module renderer_module popl segmentation rotation_math openvslam ${OpenCV_LIBS})
target_include_directories(eval_one PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

