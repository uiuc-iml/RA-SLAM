#include "utils/tsdf/voxel_types.cuh"

__device__ __host__ VoxelRGBW::VoxelRGBW() : rgb(0, 0, 0), weight(0) {}
__device__ __host__ VoxelRGBW::VoxelRGBW(const Vector3<unsigned char>& rgb, unsigned char weight)
    : rgb(rgb), weight(weight) {}

__device__ __host__ VoxelTSDF::VoxelTSDF() : tsdf(1.) {}
__device__ __host__ VoxelTSDF::VoxelTSDF(float tsdf) : tsdf(tsdf) {}

__device__ __host__ VoxelSEGM::VoxelSEGM() : probability(0.) {}
__device__ __host__ VoxelSEGM::VoxelSEGM(float probability) : probability(probability) {}

__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(){};
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Vector3<float>& position)
    : VoxelSpatialTSDF(position, 1.) {}
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Vector3<float>& position, float tsdf)
    : position(position), tsdf(tsdf) {}
