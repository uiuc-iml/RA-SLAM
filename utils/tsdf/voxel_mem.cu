#include "utils/tsdf/voxel_mem.cuh"

#include <cstdio>

__global__ static void heap_init(int *heap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < NUM_BLOCK) {
    heap[idx] = idx;
  }
}

__device__ __host__ Vector3<short> point2block(const Vector3<short> &point) {
  return point >> BLOCK_LEN_BITS;
}

__device__ __host__ Vector3<short> point2offset(const Vector3<short> &point) {
  return point & (BLOCK_LEN - 1);
}

__device__ __host__ unsigned int offset2index(const Vector3<short> &point_offset) {
  return point_offset.x + point_offset.y * BLOCK_LEN + point_offset.z * BLOCK_AREA;
}

VoxelMemPool::VoxelMemPool() {
  // initialize free block counter
  cudaMallocManaged(&num_free_blocks_, sizeof(int));
  *num_free_blocks_ = NUM_BLOCK;
  // initialize heap array
  cudaMalloc(&heap_, sizeof(int) * NUM_BLOCK);
  heap_init<<<NUM_BLOCK / 256, 256>>>(heap_);
  // initialize voxels
  cudaMalloc(&voxels_, sizeof(Voxel) * NUM_BLOCK * BLOCK_VOLUME);
  cudaMemset(voxels_, 0, sizeof(Voxel) * NUM_BLOCK * BLOCK_VOLUME);
  cudaDeviceSynchronize();
}

void VoxelMemPool::ReleaseMemory() {
  cudaFree(num_free_blocks_);
  cudaFree(heap_);
  cudaFree(voxels_);
}

__device__ Voxel* VoxelMemPool::AquireBlock() {
  const int idx = atomicSub(num_free_blocks_, 1);
  assert(idx >= 1);

  const int block_idx = heap_[idx - 1];

  return &voxels_[block_idx << BLOCK_VOLUME_BITS]; 
}

__device__ void VoxelMemPool::ReleaseBlock(const Voxel *voxel_block) {
  const int idx = atomicAdd(num_free_blocks_, 1);
  assert(idx < NUM_BLOCK);

  const int voxel_idx = (int)(voxel_block - voxels_);
  assert((voxel_idx & ((1 << BLOCK_VOLUME_BITS) - 1)) == 0);

  heap_[idx] = voxel_idx >> BLOCK_VOLUME_BITS;
}

