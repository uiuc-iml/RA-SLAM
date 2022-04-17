#include "utils/tsdf/voxel_types.cuh"

__device__ __host__ VoxelRGBW::VoxelRGBW() : rgb(0, 0, 0), weight(0) {}
__device__ __host__ VoxelRGBW::VoxelRGBW(const Eigen::Matrix<unsigned char, 3, 1>& rgb,
                                         unsigned char weight)
    : rgb(rgb), weight(weight) {}

__device__ __host__ VoxelTSDF::VoxelTSDF() : tsdf(-10) {}
__device__ __host__ VoxelTSDF::VoxelTSDF(float tsdf) : tsdf(tsdf) {}

__device__ __host__ VoxelSEGM::VoxelSEGM() {
    #pragma unroll
    for (int i = 0; i < NUM_CLASSES; ++i) {
        prob_vec[i] = 1. / NUM_CLASSES;
    }
}
__device__ __host__ VoxelSEGM::VoxelSEGM(float prob_vec_[NUM_CLASSES]) {
    #pragma unroll
    for (int i = 0; i < NUM_CLASSES; ++i) {
        prob_vec[i] = prob_vec_[i];
    }
}

__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(){};
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position)
    : VoxelSpatialTSDF(position, 1.) {}
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position, float tsdf)
    : position(position), tsdf(tsdf) {}

__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(){};
__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(const Eigen::Vector3f& position,
                                                               const float tsdf, const float prob_vec_[NUM_CLASSES])
    : position(position), tsdf(tsdf) {
        #pragma unroll
        for (int i = 0; i < NUM_CLASSES; ++i) {
            prob_vec[i] = prob_vec_[i];
        }
    }