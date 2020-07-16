#pragma once

#include <cassert>

#include "utils/tsdf/voxel_types.cuh"
#include "utils/cuda/vector.cuh"

#define NUM_BLOCK_BITS    18
#define NUM_BLOCK         (1 << NUM_BLOCK_BITS)

#define BLOCK_LEN_BITS    3
#define BLOCK_AREA_BITS   (BLOCK_LEN_BITS * 2)
#define BLOCK_VOLUME_BITS (BLOCK_LEN_BITS * 3) 
#define BLOCK_LEN         (1 << BLOCK_LEN_BITS)
#define BLOCK_AREA        (1 << BLOCK_AREA_BITS)
#define BLOCK_VOLUME      (1 << BLOCK_VOLUME_BITS)

__device__ __host__ Vector3<short> point2block(const Vector3<short> &point);

__device__ __host__ Vector3<short> point2offset(const Vector3<short> &point);

__device__ __host__ unsigned int offset2index(const Vector3<short> &point_offset);

class VoxelMemPool {
 public:
  VoxelMemPool();

  void Reset();

  void ReleaseMemory();

  __device__ Voxel* AquireBlock();

  __device__ void ReleaseBlock(const Voxel *voxel_block);

 protected:
  int *num_free_blocks_;
  int *heap_;
  Voxel *voxels_;
};

