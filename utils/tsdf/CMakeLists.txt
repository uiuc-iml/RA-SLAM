add_library(tsdf_utils voxel_types.cu voxel_tsdf.cu voxel_hash.cu voxel_mem.cu)
set_target_properties(tsdf_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tsdf_utils PUBLIC opencv_core gl_utils spdlog)

