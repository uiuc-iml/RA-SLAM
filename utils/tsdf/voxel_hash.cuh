#pragma once

#include "utils/cuda/vector.cuh"
#include "utils/tsdf/voxel_mem.cuh"
#include "utils/tsdf/voxel_types.h"

#define NUM_BUCKET_BITS           21
#define NUM_BUCKET                (1 << NUM_BUCKET_BITS)
#define BUCKET_MASK               (NUM_BUCKET - 1)

#define NUM_ENTRY_PER_BUCKET_BITS 1
#define NUM_ENTRY_PER_BUCKET      (1 << NUM_ENTRY_PER_BUCKET_BITS)
#define ENTRY_PER_BUCKET_MASK     (NUM_ENTRY_PER_BUCKET - 1)

#define NUM_ENTRY_BITS            (NUM_BUCKET_BITS + NUM_ENTRY_PER_BUCKET_BITS) 
#define NUM_ENTRY                 (1 << NUM_ENTRY_BITS)
#define ENTRY_MASK                (NUM_ENTRY - 1)

struct HashEntry {
  Vector3<short>  block_pos;
  // offset < 0:  block not in hash table
  // offset = 0:  normal entry
  // offset > 0:  part of a list
  short           offset;
  Voxel*          voxels;
};

enum BucketLock {
  FREE = 0,
  LOCKED = 1,
};

__device__ __host__ uint hash(const Vector3<short> &block_pos);

class VoxelBlock : public HashEntry {
 public:
  __device__ __host__ VoxelBlock();

  __device__ __host__ Voxel GetVoxel(const Vector3<short> &point) const;

  __device__ __host__ Voxel GetVoxel(const short idx) const;

  __device__ __host__ Voxel& GetVoxelMutable(const Vector3<short> &point) const;

  __device__ __host__ Voxel& GetVoxelMutable(const short idx) const;
};

class VoxelHashTable : public VoxelMemPool {
 public:
  VoxelHashTable();

  void ResetLocks();

  void ReleaseMemory();

  __device__ void Allocate(const Vector3<short> &block_pos);

  __device__ void Delete(const Vector3<short> &block_pos);

  __device__ Voxel Retrieve(const Vector3<short> &point, VoxelBlock &cache) const;

  __device__ Voxel* RetrieveMutable(const Vector3<short> &point, VoxelBlock &cache) const;

  __device__ __host__ int NumActiveBlock() const;

 private:
  VoxelBlock *hash_table_;
  int *bucket_locks_;
};
