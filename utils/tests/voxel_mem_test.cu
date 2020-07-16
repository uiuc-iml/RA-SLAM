#include <gtest/gtest.h>
#include <ctime>

#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_mem.cuh"

class VoxelMemTest : public ::testing::Test {
 protected:
  ~VoxelMemTest() {
    voxel_mem_pool.ReleaseMemory();
  }

  VoxelMemPool voxel_mem_pool;
};

__global__ void AquireBlocks(VoxelMemPool voxel_mem_pool, Voxel **voxel_blocks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel_blocks[idx] = voxel_mem_pool.AquireBlock();
}

__global__ void AssignBlocks(Voxel **voxel_blocks, int num_blocks) {
  const Vector3<short> thread_pos(threadIdx.x, threadIdx.y, threadIdx.z);
  const int idx = offset2index(thread_pos);
  for (int i = 0; i < num_blocks; ++i) {
    Voxel *voxel_block = voxel_blocks[i];
    voxel_block[idx].weight = i;
  }
}

__global__ void ReleaseBlocks(VoxelMemPool voxel_mem_pool, Voxel **voxel_blocks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel_mem_pool.ReleaseBlock(voxel_blocks[idx]);
}

constexpr int NUM_BLOCK_ALLOC = 8;

TEST_F(VoxelMemTest, Test1) {
  Voxel **voxel_blocks;
  // allocation
  CUDA_SAFE_CALL(cudaMallocManaged(&voxel_blocks, sizeof(Voxel*) * NUM_BLOCK_ALLOC));
  CUDA_SAFE_CALL(cudaMemset(voxel_blocks, 0, sizeof(Voxel*) * NUM_BLOCK_ALLOC));
  AquireBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_TRUE(voxel_blocks[0] != NULL);
  for (int i = 1; i < NUM_BLOCK_ALLOC; ++i) {
    EXPECT_TRUE(voxel_blocks[i] != NULL);
    EXPECT_TRUE(voxel_blocks[i-1] != voxel_blocks[i]);
  }
  // assignment
  const dim3 THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  AssignBlocks<<<1, THREAD_DIM>>>(voxel_blocks, NUM_BLOCK_ALLOC);
  CUDA_CHECK_ERROR;
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(
      cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_EQ(voxels[j].weight, i);
    }
  }
  // release (does not clobber memory)
  ReleaseBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks);
  CUDA_CHECK_ERROR;
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(
      cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_EQ(voxels[j].weight, i);
    }
  }
  // aquire again should clear voxel memory
  AquireBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(
      cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_TRUE(voxels[j].weight == 0);
    }
  }
}

