add_executable(sens_reader_manual_test sens_reader_manual_test.cc)
target_link_libraries(sens_reader_manual_test cuda_util ${OpenCV_LIBS} scannet_sens_reader)

add_executable(eval_one eval_one.cc)
target_link_libraries(eval_one
    PRIVATE gl_utils tsdf_module renderer_module popl segmentation rotation_math openvslam ${OpenCV_LIBS})
target_include_directories(eval_one PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})