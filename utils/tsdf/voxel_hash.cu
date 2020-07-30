#include <cassert>

#include "utils/cuda/arithmetic.cuh"
#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_hash.cuh"

__global__ static void reset_locks_kernel(int *locks, int num_locks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_locks) {
    locks[idx] = FREE;
  }
}

__device__ __host__ uint hash(const Vector3<short> &block_pos) {
  return (((uint)block_pos.x * 73856093u) ^
          ((uint)block_pos.y * 19349669u) ^
          ((uint)block_pos.z * 83492791u)) & BUCKET_MASK; 
}

__device__ __host__ VoxelBlock::VoxelBlock() {
  this->offset = 0;
  this->voxels = NULL;
}

__device__ __host__ Voxel VoxelBlock::GetVoxel(const Vector3<short> &point) const {
  return GetVoxelMutable(point);
}

__device__ __host__ Voxel VoxelBlock::GetVoxel(const short idx) const {
  return GetVoxelMutable(idx);
}

__device__ __host__ Voxel& VoxelBlock::GetVoxelMutable(const Vector3<short> &point) const {
  assert(voxels != NULL);
  assert(point2block(point) == this->block_pos);
  const Vector3<short> offset = point2offset(point);
  const unsigned short idx = offset2index(offset); 
  return voxels[idx];
}

__device__ __host__ Voxel& VoxelBlock::GetVoxelMutable(const short idx) const {
  assert(voxels != NULL);
  assert(idx < BLOCK_VOLUME);
  return voxels[idx];
}

VoxelHashTable::VoxelHashTable() : VoxelMemPool() {
  // initialize hash table
  CUDA_SAFE_CALL(cudaMalloc(&hash_table_, sizeof(VoxelBlock) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMemset(hash_table_, 0, sizeof(VoxelBlock) * NUM_ENTRY));
  // initialize bucket locks
  CUDA_SAFE_CALL(cudaMalloc(&bucket_locks_, sizeof(BucketLock) * NUM_BUCKET));
  CUDA_SAFE_CALL(cudaMemset(bucket_locks_, FREE, sizeof(BucketLock) * NUM_BUCKET));
}

void VoxelHashTable::ResetLocks(cudaStream_t stream) {
  reset_locks_kernel<<<NUM_BUCKET / 1024, 1024, 0, stream>>>(bucket_locks_, NUM_BUCKET);
  CUDA_STREAM_CHECK_ERROR(stream);
}

void VoxelHashTable::ReleaseMemory() {
  CUDA_SAFE_CALL(cudaFree(hash_table_));
  CUDA_SAFE_CALL(cudaFree(bucket_locks_));
  VoxelMemPool::ReleaseMemory();
}

__device__ void VoxelHashTable::Allocate(const Vector3<short> &block_pos) {
  const unsigned int bucket_idx = hash(block_pos);
  const unsigned int entry_idx = (bucket_idx << NUM_ENTRY_PER_BUCKET_BITS);
  // check for current bucket
  #pragma unroll
  for(int i = 0; i < NUM_ENTRY_PER_BUCKET; ++i) {
    VoxelBlock &block = hash_table_[entry_idx + i];
    if (block.block_pos == block_pos && block.voxels) { return; }
    if (block.voxels == NULL) {
      if (atomicExch(&bucket_locks_[bucket_idx], LOCKED) == FREE) { // lock bucket
        block.block_pos = block_pos;
        block.offset = 0;
        block.voxels = this->AquireBlock();
      }
      return;
    }
  }
  // traverse list
  unsigned int entry_idx_last = entry_idx + NUM_ENTRY_PER_BUCKET - 1;
  while (hash_table_[entry_idx_last].offset) {
    entry_idx_last = (entry_idx_last + hash_table_[entry_idx_last].offset) & ENTRY_MASK;
    const VoxelBlock &block = hash_table_[entry_idx_last];
    if (block.block_pos == block_pos && block.voxels) { return; }
  }
  const unsigned int bucket_idx_last = entry_idx_last >> NUM_ENTRY_PER_BUCKET_BITS;
  // append to list
  unsigned int entry_idx_next = entry_idx_last;
  while (true) {
    entry_idx_next = (entry_idx_next + 1) & ENTRY_MASK;
    // not last position of the bucket && hash entry empty
    if ((entry_idx_next & ENTRY_PER_BUCKET_MASK) != ENTRY_PER_BUCKET_MASK &&
        hash_table_[entry_idx_next].voxels == NULL) { 
      if (atomicExch(&bucket_locks_[bucket_idx_last], LOCKED) == FREE) { // lock last bucket
        const unsigned int bucket_idx_next = entry_idx_next >> NUM_ENTRY_PER_BUCKET_BITS;
        if (atomicExch(&bucket_locks_[bucket_idx_next], LOCKED) == FREE) { // lock next bucket
          VoxelBlock &block_last = hash_table_[entry_idx_last];
          VoxelBlock &block_next = hash_table_[entry_idx_next];
          // link new node to previous list tail
          const unsigned int wrap = entry_idx_next > entry_idx_last ? 0 : NUM_ENTRY;
          block_last.offset = entry_idx_next + wrap - entry_idx_last;
          // allocate new hash entry
          block_next.block_pos = block_pos;
          block_next.offset = 0;
          block_next.voxels = this->AquireBlock();
        }
      }
      return;
    }
  }
}

__device__ void VoxelHashTable::Delete(const Vector3<short> &block_pos) {
  const unsigned int bucket_idx = hash(block_pos);
  const unsigned int entry_idx = (bucket_idx << NUM_ENTRY_PER_BUCKET_BITS);
  // check for current bucket
  #pragma unroll
  for(int i = 0; i < NUM_ENTRY_PER_BUCKET - 1; ++i) {
    VoxelBlock &block = hash_table_[entry_idx + i];
    if (block.block_pos == block_pos && block.voxels) { 
      this->ReleaseBlock(block.voxels);
      block.offset = 0;
      block.voxels = NULL;
      return; 
    }
  }
  // special handling for list head
  unsigned int entry_idx_last = entry_idx + NUM_ENTRY_PER_BUCKET - 1;
  VoxelBlock &block_head = hash_table_[entry_idx_last];
  if (block_head.block_pos == block_pos && block_head.voxels) {
    if (atomicExch(&bucket_locks_[bucket_idx], LOCKED) == FREE) {
      const unsigned int entry_idx_next = (entry_idx_last + block_head.offset) & ENTRY_MASK;
      VoxelBlock &block_next = hash_table_[entry_idx_next];
      this->ReleaseBlock(block_head.voxels);
      block_head.block_pos = block_next.block_pos;
      // check if reaches tail
      block_head.offset = block_next.offset ? block_head.offset + block_next.offset : 0;
      block_head.voxels = block_next.voxels;
      block_next.offset = 0;
      block_next.voxels = NULL;
    }
    return;
  }
  // generic list handling
  while (hash_table_[entry_idx_last].offset) {
    VoxelBlock &block_last = hash_table_[entry_idx_last];
    const unsigned int entry_idx_curr = (entry_idx_last + block_last.offset) & ENTRY_MASK;
    VoxelBlock &block_curr = hash_table_[entry_idx_curr];
    if (block_curr.block_pos == block_pos && block_curr.voxels) {
      if (atomicExch(&bucket_locks_[bucket_idx], LOCKED) == FREE) { // lock original bucket
        // check if reaches tail
        block_last.offset = block_curr.offset ? block_last.offset + block_curr.offset : 0;
        // free current entry
        this->ReleaseBlock(block_curr.voxels);
        block_curr.offset = 0;
        block_curr.voxels = NULL;
      }
      return;
    }
    entry_idx_last = entry_idx_curr;
  }
}

__device__ float VoxelHashTable::RetrieveTSDF(const Vector3<float> &point, 
                                              VoxelBlock &cache) const {
  const Vector3<float> point_l = point.cast<short>().cast<float>();
  const Vector3<float> point_h = point_l + 1;
  const Vector3<float> alpha = point_h - point;
  const float tsdf_000 = Retrieve(Vector3<short>(point_l.x, point_l.y, point_l.z), cache).tsdf;
  const float tsdf_001 = Retrieve(Vector3<short>(point_h.x, point_l.y, point_h.z), cache).tsdf;
  const float tsdf_010 = Retrieve(Vector3<short>(point_l.x, point_h.y, point_l.z), cache).tsdf;
  const float tsdf_011 = Retrieve(Vector3<short>(point_l.x, point_h.y, point_h.z), cache).tsdf;
  const float tsdf_100 = Retrieve(Vector3<short>(point_h.x, point_l.y, point_l.z), cache).tsdf;
  const float tsdf_101 = Retrieve(Vector3<short>(point_h.x, point_l.y, point_h.z), cache).tsdf;
  const float tsdf_110 = Retrieve(Vector3<short>(point_h.x, point_h.y, point_l.z), cache).tsdf;
  const float tsdf_111 = Retrieve(Vector3<short>(point_h.x, point_h.y, point_h.z), cache).tsdf;
  // interpolate across z
  const float tsdf_00 = tsdf_000 * alpha.z + tsdf_001 * (1 - alpha.z);
  const float tsdf_01 = tsdf_010 * alpha.z + tsdf_011 * (1 - alpha.z);
  const float tsdf_10 = tsdf_100 * alpha.z + tsdf_101 * (1 - alpha.z);
  const float tsdf_11 = tsdf_110 * alpha.z + tsdf_111 * (1 - alpha.z);
  // interpolate across y
  const float tsdf_0 = tsdf_00 * alpha.y + tsdf_01 * (1 - alpha.y);
  const float tsdf_1 = tsdf_10 * alpha.y + tsdf_11 * (1 - alpha.y);
  // interpolate across x
  return tsdf_0 * alpha.x + tsdf_1 * (1 - alpha.x);
}

__device__ Voxel VoxelHashTable::Retrieve(const Vector3<short> &point, VoxelBlock &cache) const {
  Voxel *voxel = RetrieveMutable(point, cache);
  if (voxel)
    return *voxel;
  // not found -> empty space
  return { 1., { 0, 0, 0 }, 0 };
}

__device__ Voxel* VoxelHashTable::RetrieveMutable(const Vector3<short> &point, 
                                                  VoxelBlock &cache) const {
  const Vector3<short> block_pos = point2block(point);
  if (cache.block_pos == block_pos) {
    if (cache.voxels) {
      return &(cache.GetVoxelMutable(point));
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
    if (block.block_pos == block_pos && block.voxels) { 
      cache = block;
      return &(cache.GetVoxelMutable(point));
    }
  }
  // traverse list
  unsigned int entry_idx_last = entry_idx + NUM_ENTRY_PER_BUCKET - 1;
  while (hash_table_[entry_idx_last].offset) {
    entry_idx_last = (entry_idx_last + hash_table_[entry_idx_last].offset) & ENTRY_MASK;
    const VoxelBlock &block = hash_table_[entry_idx_last];
    if (block.block_pos == block_pos && block.voxels) { 
      cache = block;
      return &(cache.GetVoxelMutable(point));
    }
  }
  // not found
  cache.block_pos = block_pos;
  cache.offset = -1;
  cache.voxels = NULL;
  return NULL;
}

__device__ const VoxelBlock& VoxelHashTable::GetBlock(const int idx) const {
  assert(idx < NUM_ENTRY);
  return hash_table_[idx];
}

__device__ __host__ int VoxelHashTable::NumActiveBlock() const {
  return NUM_BLOCK - *num_free_blocks_;
}
