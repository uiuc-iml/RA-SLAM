add_library(utils stereo_rectifier.cc)
target_link_libraries(utils PRIVATE opencv_imgproc opencv_calib3d openvslam::openvslam)

add_subdirectory(gl)
add_subdirectory(tsdf)

# Tests
add_executable(voxel_mem_test tests/voxel_mem_test.cu)
target_link_libraries(voxel_mem_test tsdf gtest gtest_main)
set_target_properties(voxel_mem_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(voxel_mem_test)

add_executable(voxel_hash_test tests/voxel_hash_test.cu)
target_link_libraries(voxel_hash_test tsdf gtest gtest_main)
set_target_properties(voxel_hash_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(voxel_hash_test)
