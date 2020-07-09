add_library(tsdf voxel_tsdf.cu voxel_hash.cu voxel_mem.cu)
set_target_properties(tsdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


