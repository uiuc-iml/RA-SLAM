#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>

#define NUM_CLASSES 21  // multi-class segmentation

struct MultiClsSemantics {
  int max_cls;
  int observation_cnt;

  __device__ __host__ MultiClsSemantics();

  __device__ __host__ void uniform_init();

  __device__ int get_max_class() const;

  __device__ void update(const float* prob_map, const int img_idx, const int class_offset);
};

/**
 * @brief voxel data packed with RGB and TSDF weight
 */
class VoxelRGBW {
 public:
  Eigen::Matrix<unsigned char, 3, 1> rgb;
  unsigned char weight;

 public:
  __device__ __host__ VoxelRGBW();
  __device__ __host__ VoxelRGBW(const Eigen::Matrix<unsigned char, 3, 1>& rgb,
                                const unsigned char weight);
};

/**
 * @brief voxel data of tsdf value
 */
class VoxelTSDF {
 public:
  float tsdf;

 public:
  __device__ __host__ VoxelTSDF();
  __device__ __host__ VoxelTSDF(float tsdf);
};

/**
 * @brief voxel data of segmentation probabilities
 */
class VoxelSEGM {
 public:
  MultiClsSemantics semantic_rep;

 public:
  __device__ __host__ VoxelSEGM();
};

/**
 * @brief voxel data of spatial location and tsdf value
 */
class VoxelSpatialTSDF {
 public:
  Eigen::Vector3f position;
  float tsdf;

 public:
  __device__ __host__ VoxelSpatialTSDF();
  __device__ __host__ VoxelSpatialTSDF(const Eigen::Vector3f& position);
  __device__ __host__ VoxelSpatialTSDF(const Eigen::Vector3f& position, float tsdf);
};

/**
 * @brief voxel data of spatial location and tsdf value
 */
class VoxelSpatialTSDFSEGM {
 public:
  Eigen::Vector3f position;
  float tsdf;
  int predicted_class;

 public:
  __device__ __host__ VoxelSpatialTSDFSEGM();
  __device__ __host__ VoxelSpatialTSDFSEGM(const Eigen::Vector3f& position, const float tsdf,
                                           const int predicted_class_);
};