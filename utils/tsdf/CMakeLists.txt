add_library(tsdf_utils voxel_types.cu voxel_tsdf.cu voxel_hash.cu voxel_mem.cu)
target_link_libraries(tsdf_utils PUBLIC opencv_core gl_utils Eigen3::Eigen
                                 PRIVATE openvslam cuda_util)

