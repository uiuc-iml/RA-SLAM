#pragma once

#include <cassert>
#include <cuda_runtime.h>

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

struct VoxelBlock {
  Vector3<short>  position;
  // offset < 0:  block not in hash table
  // offset = 0:  normal entry
  // offset > 0:  part of a list
  short           offset;
  int             idx;

  __device__ __host__ VoxelBlock() : offset(0), idx(-1) {};

  __device__ __host__ VoxelBlock(int idx) : offset(0), idx(idx) {};
};

class VoxelMemPool {
 public:
  VoxelMemPool();

  void Reset();

  void ReleaseMemory();

  __device__ int AquireBlock();

  __device__ void ReleaseBlock(const int block_idx);

  template<typename Voxel>
  __device__ Voxel& GetVoxel(const Vector3<short> &point, const VoxelBlock &block) const {
    assert(block.idx >= 0 && block.idx < NUM_BLOCK);
    assert(point2block(point) == block.position);
    const Vector3<short> offset = point2offset(point);
    const unsigned short idx = offset2index(offset);
    return GetVoxel<Voxel>(idx, block);
  }

  template<typename Voxel>
  __device__ Voxel& GetVoxel(const int idx, const VoxelBlock &block) const {
    assert(idx >= 0 && idx < BLOCK_VOLUME);
    Voxel *voxels = GetVoxelData<Voxel>();
    return voxels[(block.idx << BLOCK_VOLUME_BITS) + idx];
  }

  __host__ int NumFreeBlocks() const;

 private:
  template <typename Voxel>
  __device__ constexpr Voxel* GetVoxelData() const;

  VoxelRGBW *voxels_rgbw_;
  VoxelTSDF *voxels_tsdf_;
  VoxelSEGM *voxels_segm_;
  int *num_free_blocks_;
  int *heap_;
};

template <>
__device__ inline VoxelRGBW* VoxelMemPool::GetVoxelData<VoxelRGBW>() const {
  return voxels_rgbw_;
}

template <>
__device__ inline VoxelTSDF* VoxelMemPool::GetVoxelData<VoxelTSDF>() const {
  return voxels_tsdf_;
}

template <>
__device__ inline VoxelSEGM* VoxelMemPool::GetVoxelData<VoxelSEGM>() const {
  return voxels_segm_;
}

