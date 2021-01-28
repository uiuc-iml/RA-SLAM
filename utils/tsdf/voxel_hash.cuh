#pragma once

#include <cassert>
#include <cuda_runtime.h>

#include "utils/cuda/vector.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/tsdf/voxel_types.cuh"
#include "utils/tsdf/voxel_mem.cuh"

// number of buckets in the hash table
#define NUM_BUCKET_BITS           21
#define NUM_BUCKET                (1 << NUM_BUCKET_BITS)
#define BUCKET_MASK               (NUM_BUCKET - 1)

// number of entries per bucket
#define NUM_ENTRY_PER_BUCKET_BITS 1
#define NUM_ENTRY_PER_BUCKET      (1 << NUM_ENTRY_PER_BUCKET_BITS)
#define ENTRY_PER_BUCKET_MASK     (NUM_ENTRY_PER_BUCKET - 1)

// number of entries in the hash table
#define NUM_ENTRY_BITS            (NUM_BUCKET_BITS + NUM_ENTRY_PER_BUCKET_BITS)
#define NUM_ENTRY                 (1 << NUM_ENTRY_BITS)
#define ENTRY_MASK                (NUM_ENTRY - 1)

/**
 * @brief hash function for a 3D index
 *
 * @param block_pos 3D voxel block position (on integer grid)
 *
 * @return  hashed index
 */
__device__ __host__ uint Hash(const Vector3<short> &block_pos);

/**
 * @brief bucket lock state
 */
enum BucketLock {
  FREE = 0,
  LOCKED = 1,
};

/**
 * @brief abstraction for a hash data structure in GPU memory
 */
class VoxelHashTable {
 public:
  /**
   * @brief internal GPU memory allocation
   */
  VoxelHashTable();

  /**
   * @brief (de)allocation causes certain bucket to be locked, this frees all locks
   *
   * @param stream optional CUDA stream
   */
  void ResetLocks(cudaStream_t stream = NULL);

  /**
   * @brief release all internal GPU memory
   */
  void ReleaseMemory();

  /**
   * @brief allocate a block of voxel into the hash table
   *
   * @param block_pos voxel block position
   */
  __device__ void Allocate(const Vector3<short> &block_pos);

  /**
   * @brief deallocate a block of voxel from the hash table
   *
   * @param block_pos voxel block position
   */
  __device__ void Delete(const Vector3<short> &block_pos);

  /**
   * @brief get tsdf value of a voxel using bilinear interpolation through adjacent voxels
   *
   * @param point location of the voxel in voxel grid coordinate
   * @param cache cached voxel block meta data for accelerating sequential voxel access
   *              within the same voxel block
   *
   * @return tsdf value of a point
   */
  __device__ float RetrieveTSDF(const Vector3<float> &point, VoxelBlock &cache) const;

  /**
   * @brief get voxel data at integer voxel coordinate
   *
   * @tparam Voxel  voxel type
   * @param point   location of the voxel in voxel grid coordinate
   * @param cache   cached voxel block meta data for accelerating sequential voxel access
   *                within the same voxel block
   *
   * @return voxel data in Voxel type (default voxel if voxel not found in hash table)
   */
  template <typename Voxel>
  __device__ Voxel Retrieve(const Vector3<short> &point, VoxelBlock &cache) const {
    Voxel *voxel = RetrieveMutable<Voxel>(point, cache);
    if (voxel) { return *voxel; }
    else { return Voxel(); }
  }

  /**
   * @brief get voxel mutable data at integer voxel coordinate
   *
   * @tparam Voxel  voxel type
   * @param point   location of the voxel in voxel grid coordinate
   * @param cache   cached voxel block meta data for accelerating sequential voxel access
   *                within the same voxel block
   *
   * @return voxel data in Voxel type (nullptr if voxel not found in hash table)
   */
  template <typename Voxel>
  __device__ Voxel* RetrieveMutable(const Vector3<short> &point, VoxelBlock &cache) const {
    const Vector3<short> block_pos = PointToBlock(point);
    if (cache.position == block_pos) {
      if (cache.idx >= 0) {
        return &(mem.GetVoxel<Voxel>(point, cache));
      }
      else if (cache.offset < 0) {
        return nullptr;
      }
    }
    const unsigned int bucket_idx = Hash(block_pos);
    const unsigned int entry_idx = (bucket_idx << NUM_ENTRY_PER_BUCKET_BITS);
    // check for current bucket
    #pragma unroll
    for(int i = 0; i < NUM_ENTRY_PER_BUCKET; ++i) {
      VoxelBlock &block = hash_table_[entry_idx + i];
      if (block.position == block_pos && block.idx >= 0) {
        cache = block;
        return &(mem.GetVoxel<Voxel>(point, cache));
      }
    }
    // traverse list
    unsigned int entry_idx_last = entry_idx + NUM_ENTRY_PER_BUCKET - 1;
    while (hash_table_[entry_idx_last].offset) {
      entry_idx_last = (entry_idx_last + hash_table_[entry_idx_last].offset) & ENTRY_MASK;
      const VoxelBlock &block = hash_table_[entry_idx_last];
      if (block.position == block_pos && block.idx >= 0) {
        cache = block;
        return &(mem.GetVoxel<Voxel>(point, cache));
      }
    }
    // not found
    cache.position = block_pos;
    cache.offset = -1;
    cache.idx = -1;
    return NULL;
  }

  /**
   * @brief get voxel block given hash table index
   *
   * @param idx index into the hash table
   *
   * @return  voxel block meta data
   */
  __device__ const VoxelBlock& GetBlock(const int idx) const;

  /**
   * @return number of active voxel blocks
   */
  __host__ int NumActiveBlock() const;

 public:
  VoxelMemPool mem;

 private:
  VoxelBlock *hash_table_;
  int *bucket_locks_;
};
