#include <gtest/gtest.h>
#include <ctime>

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

__global__ void AquireRead(VoxelMemPool voxel_mem_pool, Voxel **voxel_blocks, int *indics) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel_blocks[idx] = voxel_mem_pool.AquireBlock();
  indics[idx] = voxel_blocks[idx][0].weight;
}

constexpr int NUM_BLOCK_ALLOC = 8;

TEST_F(VoxelMemTest, Test1) {
  Voxel **voxel_blocks;
  // allocation
  cudaMallocManaged(&voxel_blocks, sizeof(Voxel*) * NUM_BLOCK_ALLOC);
  cudaMemset(voxel_blocks, 0, sizeof(Voxel*) * NUM_BLOCK_ALLOC);
  AquireBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks);
  cudaDeviceSynchronize();
  EXPECT_TRUE(voxel_blocks[0] != NULL);
  for (int i = 1; i < NUM_BLOCK_ALLOC; ++i) {
    EXPECT_TRUE(voxel_blocks[i] != NULL);
    EXPECT_TRUE(voxel_blocks[i-1] != voxel_blocks[i]);
  }
  // assignment
  AssignBlocks<<<1, dim3(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN)>>>(voxel_blocks, NUM_BLOCK_ALLOC);
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost);
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_TRUE(voxels[j].weight == i);
    }
  }
  // release (does not clobber memory)
  ReleaseBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks);
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost);
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_TRUE(voxels[j].weight == i);
    }
  }
  // aquire without assignment should get the same set of blocks (with arbitrary order)
  int *indics;
  cudaMallocManaged(&indics, sizeof(int) * NUM_BLOCK_ALLOC);
  AquireRead<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks, indics);
  cudaDeviceSynchronize(); 
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    Voxel voxels[BLOCK_VOLUME];
    cudaMemcpy(voxels, voxel_blocks[i], sizeof(Voxel) * BLOCK_VOLUME, cudaMemcpyDeviceToHost);
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_TRUE(voxels[j].weight == indics[i]);
    }
  }
}

