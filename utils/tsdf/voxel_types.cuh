#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include <cuda_fp16.h>

#define NUM_CLASSES 21 // multi-class segmentation

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
  __half prob_vec[NUM_CLASSES];

 public:
  __device__ __host__ VoxelSEGM();
  __device__ __host__ VoxelSEGM(__half prob_vec_[NUM_CLASSES]);
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
  __device__ __host__ VoxelSpatialTSDFSEGM(const Eigen::Vector3f& position, const float tsdf, const int predicted_class_);
};