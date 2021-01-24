#pragma once

#include <cassert>
#include <cuda_runtime.h>

#include "utils/cuda/vector.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/tsdf/voxel_types.cuh"
#include "utils/tsdf/voxel_mem.cuh"

#define NUM_BUCKET_BITS           21
#define NUM_BUCKET                (1 << NUM_BUCKET_BITS)
#define BUCKET_MASK               (NUM_BUCKET - 1)

#define NUM_ENTRY_PER_BUCKET_BITS 1
#define NUM_ENTRY_PER_BUCKET      (1 << NUM_ENTRY_PER_BUCKET_BITS)
#define ENTRY_PER_BUCKET_MASK     (NUM_ENTRY_PER_BUCKET - 1)

#define NUM_ENTRY_BITS            (NUM_BUCKET_BITS + NUM_ENTRY_PER_BUCKET_BITS)
#define NUM_ENTRY                 (1 << NUM_ENTRY_BITS)
#define ENTRY_MASK                (NUM_ENTRY - 1)

__device__ __host__ uint hash(const Vector3<short> &block_pos);

enum BucketLock {
  FREE = 0,
  LOCKED = 1,
};

class VoxelHashTable {
 public:
  VoxelHashTable();

  void ResetLocks(cudaStream_t stream = NULL);

  void ReleaseMemory();

  __device__ void Allocate(const Vector3<short> &block_pos);

  __device__ void Delete(const Vector3<short> &block_pos);

  __device__ float RetrieveTSDF(const Vector3<float> &point, VoxelBlock &cache) const;

  template <typename Voxel>
  __device__ Voxel Retrieve(const Vector3<short> &point, VoxelBlock &cache) const {
    Voxel *voxel = RetrieveMutable<Voxel>(point, cache);
    if (voxel) { return *voxel; }
    else { return Voxel(); }
  }

  template <typename Voxel>
  __device__ Voxel* RetrieveMutable(const Vector3<short> &point, VoxelBlock &cache) const {
    const Vector3<short> block_pos = point2block(point);
    if (cache.position == block_pos) {
      if (cache.idx >= 0) {
        return &(mem.GetVoxel<Voxel>(point, cache));
      }
      else if (cache.offset < 0) {
        return NULL;
      }
    }
    const unsigned int bucket_idx = hash(block_pos);
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

  __device__ const VoxelBlock& GetBlock(const int idx) const;

  __host__ int NumActiveBlock() const;

 public:
  VoxelMemPool mem;

 private:
  template <typename Voxel>
  __device__ Voxel* RetrieveMutable(const Vector3<short> &point,
                                    Voxel *voxels, VoxelBlock &cache) const {
  }

  VoxelBlock *hash_table_;
  int *bucket_locks_;
};
