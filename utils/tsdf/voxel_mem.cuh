#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cassert>

#include "utils/tsdf/voxel_types.cuh"

// total number of voxel blocks available
#define NUM_BLOCK_BITS 18
// 2**18 blocks
#define NUM_BLOCK (1 << NUM_BLOCK_BITS)

// block properties
// 2^3
#define BLOCK_LEN_BITS 3
#define BLOCK_AREA_BITS (BLOCK_LEN_BITS * 2)
#define BLOCK_VOLUME_BITS (BLOCK_LEN_BITS * 3)
#define BLOCK_LEN (1 << BLOCK_LEN_BITS)         // length
#define BLOCK_AREA (1 << BLOCK_AREA_BITS)       // square
#define BLOCK_VOLUME (1 << BLOCK_VOLUME_BITS)   // cubic

__device__ __host__ inline bool Equals(
  const Eigen::Matrix<short, 3, 1>& a,
  const Eigen::Matrix<short, 3, 1>& b
) {
  return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

/**
 * @brief convert voxel coordinate to block coordinate
 *
 * @param point voxel coordinate in integer grid
 *
 * @return block cooridnate in integer grid
 */
__device__ __host__ inline Eigen::Matrix<short, 3, 1> PointToBlock(
    const Eigen::Matrix<short, 3, 1>& point) {
  return point.unaryExpr([](const short x) -> short { return x >> BLOCK_LEN_BITS; });
}

/**
 * @brief convert block coordinate to voxel coordinate
 *
 * @param block_pos block coordinate in integer grid
 *
 * @return voxel cooridnate in integer grid
 */
__device__ __host__ inline Eigen::Matrix<short, 3, 1> BlockToPoint(
    const Eigen::Matrix<short, 3, 1>& block_pos) {
  return block_pos.unaryExpr([](const short x) -> short { return x << BLOCK_LEN_BITS; });
}

/**
 * @brief convert voxel coordinate to voxel block index
 *
 * @param point voxel coordinate in integer grid
 *
 * @return relative voxel index in [0, BLOCK_VOLUME)
 */
__device__ __host__ inline Eigen::Matrix<short, 3, 1> PointToOffset(
    const Eigen::Matrix<short, 3, 1>& point) {
  return point.unaryExpr([](const short x) -> short { return x & (BLOCK_LEN - 1); });
}

/**
 * @brief convert voxel relative offset to voxel relative index
 *
 * @param point_offset  3D offset w.r.t. the first voxel in the block
 *
 * @return relative voxel index in [0, BLOCK_VOLUME)
 */
__device__ __host__ inline unsigned int OffsetToIndex(
    const Eigen::Matrix<short, 3, 1>& point_offset) {
  return point_offset[0] + point_offset[1] * BLOCK_LEN + point_offset[2] * BLOCK_AREA;
}

/**
 * @brief data structure for voxel block meta data
 */
struct VoxelBlock {
  // block position in integer grid
  Eigen::Matrix<short, 3, 1> position;
  // offset < 0:  block not in hash table (TODO(alvin): CPU streaming is not yet
  // implemented) offset = 0:  normal entry offset > 0:  part of a list
  short offset;
  // index into the memory pool of voxel blocks
  int idx;

  /**
   * @brief default construction of a non-existing block
   */
  __device__ __host__ VoxelBlock() : offset(0), idx(-1){};

  /**
   * @brief construct a block with a valid memory pool index
   *
   * @param idx index into the memory pool of voxel blocks
   */
  __device__ __host__ VoxelBlock(int idx) : offset(0), idx(idx){};
};

class VoxelMemPool {
 public:
  /**
   * @brief internal allocation of intermediate GPU buffers
   */
  VoxelMemPool();

  /**
   * @brief release all internally allocated GPU buffers
   */
  void ReleaseMemory();

  /**
   * @brief aquire a voxel block from the pre-allocated heap
   *
   * @return index of the voxel block
   */
  __device__ int AquireBlock();

  /**
   * @brief release a voxel block back to the heap
   *
   * @param block_idx index of the voxel block
   */
  __device__ void ReleaseBlock(const int block_idx);

  /**
   * @brief read voxel data from integer voxel location
   *
   * @tparam Voxel  voxel type
   * @param point   voxel position in integer grid
   * @param block   voxel block meta data
   *
   * @return  voxel data with Voxel type
   */
  template <typename Voxel>
  __device__ Voxel& GetVoxel(const Eigen::Matrix<short, 3, 1>& point,
                             const VoxelBlock& block) const {
    assert(block.idx >= 0 && block.idx < NUM_BLOCK);
    assert(Equals(PointToBlock(point), block.position));
    const Eigen::Matrix<short, 3, 1> offset = PointToOffset(point);
    const unsigned short idx = OffsetToIndex(offset);
    return GetVoxel<Voxel>(idx, block);
  }

  /**
   * @brief read voxel data with relative voxel index
   *
   * @tparam Voxel  voxel type
   * @param idx     index into a block of voxel, in [0, BLOCK_VOLUMN)
   * @param block   voxel block meta data
   *
   * @return  voxel data with Voxel type
   */
  template <typename Voxel>
  __device__ Voxel& GetVoxel(const int idx, const VoxelBlock& block) const {
    assert(idx >= 0 && idx < BLOCK_VOLUME);
    Voxel* voxels = GetVoxelData<Voxel>();
    return voxels[(block.idx << BLOCK_VOLUME_BITS) + idx];
  }

  __host__ int NumFreeBlocks() const;

 private:
  /**
   * @brief get pointer to the first voxel of this block
   *
   * @tparam Voxel  voxel type
   *
   * @return pointer to the first voxel of this block
   */
  template <typename Voxel>
  __device__ constexpr Voxel* GetVoxelData() const;

  VoxelRGBW* voxels_rgbw_;
  VoxelTSDF* voxels_tsdf_;
  VoxelSEGM* voxels_segm_;
  int* num_free_blocks_;
  int* heap_;
};

/* template specialization for each type of voxel */

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
