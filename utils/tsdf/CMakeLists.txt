add_library(tsdf voxel_hash.cu)
set_target_properties(tsdf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
