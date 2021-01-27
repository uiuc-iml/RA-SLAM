#pragma once

#include <cuda_runtime.h>
#include "utils/cuda/vector.cuh"

class VoxelRGBW {
 public:
  Vector3<unsigned char>  rgb;
  unsigned char           weight;
 public:
  __device__ __host__ VoxelRGBW();
  __device__ __host__ VoxelRGBW(const Vector3<unsigned char> &rgb, const unsigned char weight);
};

class VoxelTSDF {
 public:
  float tsdf;
 public:
  __device__ __host__ VoxelTSDF();
  __device__ __host__ VoxelTSDF(float tsdf);
};

class VoxelSEGM {
 public:
  float probability;
 public:
  __device__ __host__ VoxelSEGM();
  __device__ __host__ VoxelSEGM(float probability);
};

class VoxelSpatialTSDF {
 public:
  Vector3<float> position;
  float tsdf;
 public:
  __device__ __host__ VoxelSpatialTSDF();
  __device__ __host__ VoxelSpatialTSDF(const Vector3<float> &position);
  __device__ __host__ VoxelSpatialTSDF(const Vector3<float> &position, float tsdf);
};
