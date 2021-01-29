#pragma once

#include <cuda_runtime.h>

#include "utils/cuda/vector.cuh"

/**
 * @brief voxel data packed with RGB and TSDF weight
 */
class VoxelRGBW {
 public:
  Vector3<unsigned char> rgb;
  unsigned char weight;

 public:
  __device__ __host__ VoxelRGBW();
  __device__ __host__ VoxelRGBW(const Vector3<unsigned char>& rgb, const unsigned char weight);
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
  float probability;

 public:
  __device__ __host__ VoxelSEGM();
  __device__ __host__ VoxelSEGM(float probability);
};

/**
 * @brief voxel data of spatial location and tsdf value
 */
class VoxelSpatialTSDF {
 public:
  Vector3<float> position;
  float tsdf;

 public:
  __device__ __host__ VoxelSpatialTSDF();
  __device__ __host__ VoxelSpatialTSDF(const Vector3<float>& position);
  __device__ __host__ VoxelSpatialTSDF(const Vector3<float>& position, float tsdf);
};
