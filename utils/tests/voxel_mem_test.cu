#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <ctime>

#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_mem.cuh"

class VoxelMemTest : public ::testing::Test {
 protected:
  ~VoxelMemTest() { voxel_mem_pool.ReleaseMemory(); }

  VoxelMemPool voxel_mem_pool;
};

__global__ void AquireBlocks(VoxelMemPool voxel_mem, VoxelRGBW** voxel_blocks, int* block_indics) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  block_indics[idx] = voxel_mem.AquireBlock();
  const VoxelBlock block(block_indics[idx]);
  voxel_blocks[idx] = &voxel_mem.GetVoxel<VoxelRGBW>(0, block);
}

__global__ void AssignBlocks(VoxelMemPool voxel_mem, int* block_indics, int num_blocks) {
  const Eigen::Matrix<short, 3, 1> thread_pos(threadIdx.x, threadIdx.y, threadIdx.z);
  const int idx = OffsetToIndex(thread_pos);
  for (int i = 0; i < num_blocks; ++i) {
    const VoxelBlock block(block_indics[i]);
    VoxelRGBW& voxel = voxel_mem.GetVoxel<VoxelRGBW>(idx, block);
    voxel.weight = i;
  }
}

__global__ void ReleaseBlocks(VoxelMemPool voxel_mem, int* block_indics) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel_mem.ReleaseBlock(block_indics[idx]);
}

TEST_F(VoxelMemTest, Test1) {
  constexpr int NUM_BLOCK_ALLOC = 8;

  VoxelRGBW** voxel_blocks;
  int* block_indics;
  // allocation
  CUDA_SAFE_CALL(cudaMallocManaged(&voxel_blocks, sizeof(VoxelRGBW*) * NUM_BLOCK_ALLOC));
  CUDA_SAFE_CALL(cudaMemset(voxel_blocks, 0, sizeof(VoxelRGBW*) * NUM_BLOCK_ALLOC));
  CUDA_SAFE_CALL(cudaMallocManaged(&block_indics, sizeof(int) * NUM_BLOCK_ALLOC));
  AquireBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks, block_indics);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    EXPECT_TRUE(voxel_blocks[i] != NULL);
    if (i != 0) {
      EXPECT_TRUE(voxel_blocks[i - 1] != voxel_blocks[i]);
      EXPECT_TRUE(block_indics[i - 1] != block_indics[i]);
    }
  }
  // assignment
  const dim3 THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  AssignBlocks<<<1, THREAD_DIM>>>(voxel_mem_pool, block_indics, NUM_BLOCK_ALLOC);
  CUDA_CHECK_ERROR;
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    VoxelRGBW voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(cudaMemcpy(voxels, voxel_blocks[i], sizeof(VoxelRGBW) * BLOCK_VOLUME,
                              cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_EQ(voxels[j].weight, i);
    }
  }
  // release (does not clobber memory)
  ReleaseBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, block_indics);
  CUDA_CHECK_ERROR;
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    VoxelRGBW voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(cudaMemcpy(voxels, voxel_blocks[i], sizeof(VoxelRGBW) * BLOCK_VOLUME,
                              cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_EQ(voxels[j].weight, i);
    }
  }
  // aquire again should clear voxel memory
  AquireBlocks<<<1, NUM_BLOCK_ALLOC>>>(voxel_mem_pool, voxel_blocks, block_indics);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < NUM_BLOCK_ALLOC; ++i) {
    VoxelRGBW voxels[BLOCK_VOLUME];
    CUDA_SAFE_CALL(cudaMemcpy(voxels, voxel_blocks[i], sizeof(VoxelRGBW) * BLOCK_VOLUME,
                              cudaMemcpyDeviceToHost));
    for (int j = 0; j < BLOCK_VOLUME; ++j) {
      EXPECT_TRUE(voxels[j].weight == 1);
    }
  }
}
