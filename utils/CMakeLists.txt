add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PUBLIC opencv_imgproc opencv_calib3d yaml-cpp
                            PRIVATE openvslam::openvslam)

add_subdirectory(rotation_math)

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(gl)
    add_subdirectory(tsdf)
    add_subdirectory(tests)
endif ()

