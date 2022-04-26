#include "utils/tsdf/voxel_types.cuh"

__device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

__device__ __host__ void MultiClsSemantics::uniform_init() {
#pragma unroll
  for (int i = 0; i < NUM_CLASSES; ++i) {
    prob_vec[i] = __float2half(1. / NUM_CLASSES);
  }
}

__device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
                                          const int class_offset) {
  __half norm_coeff = __float2half(0.0f);
#pragma unroll
  for (int i = 0; i < NUM_CLASSES; ++i) {
    __half new_prob = __hmul(prob_vec[i], __float2half(prob_map[img_idx + class_offset * i]));
    prob_vec[i] = new_prob;
    norm_coeff = __hadd(norm_coeff, new_prob);
  }
#pragma unroll
  for (int i = 0; i < NUM_CLASSES; ++i) {
    prob_vec[i] = __hdiv(prob_vec[i], norm_coeff);
  }
}

__device__ int MultiClsSemantics::get_max_class() const {
  int max_cls = 0;
  __half max_prob = prob_vec[0];
#pragma unroll
  for (int i = 0; i < NUM_CLASSES; ++i) {
    if (__hgt(prob_vec[i], max_prob)) {
      max_cls = i;
      max_prob = prob_vec[i];
    }
  }
  return max_cls;
}

__device__ __host__ VoxelRGBW::VoxelRGBW() : rgb(0, 0, 0), weight(0) {}
__device__ __host__ VoxelRGBW::VoxelRGBW(const Eigen::Matrix<unsigned char, 3, 1>& rgb,
                                         unsigned char weight)
    : rgb(rgb), weight(weight) {}

__device__ __host__ VoxelTSDF::VoxelTSDF() : tsdf(-10) {}
__device__ __host__ VoxelTSDF::VoxelTSDF(float tsdf) : tsdf(tsdf) {}

__device__ __host__ VoxelSEGM::VoxelSEGM() { semantic_rep.uniform_init(); }

__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(){};
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position)
    : VoxelSpatialTSDF(position, 1.) {}
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position, float tsdf)
    : position(position), tsdf(tsdf) {}

__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(){};
__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(const Eigen::Vector3f& position,
                                                               const float tsdf,
                                                               const int predicted_class_)
    : position(position), tsdf(tsdf), predicted_class(predicted_class_) {}
