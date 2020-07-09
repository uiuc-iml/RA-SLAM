#include <cassert>
#include <gtest/gtest.h>

#include "utils/tsdf/voxel_hash.cuh"

#define MAX_BLOCKS 128

class VoxelHashTest : public ::testing::Test {
 protected:
  VoxelHashTest() {
    cudaMallocManaged(&voxel, sizeof(Voxel) * MAX_BLOCKS * BLOCK_VOLUME);
    cudaMallocManaged(&voxel_block, sizeof(VoxelBlock) * MAX_BLOCKS);
    cudaMallocManaged(&point, sizeof(Vector3<short>) * MAX_BLOCKS * BLOCK_VOLUME);
    cudaMallocManaged(&block_pos, sizeof(Vector3<short>) * MAX_BLOCKS);
  }

  ~VoxelHashTest() {
    voxel_hash_table.ReleaseMemory();
    cudaFree(voxel);
    cudaFree(voxel_block);
    cudaFree(point);
    cudaFree(block_pos);
  }

  VoxelHashTable voxel_hash_table;
  Voxel *voxel;
  VoxelBlock *voxel_block;
  Vector3<short> *point;
  Vector3<short> *block_pos;
};

__global__ void Allocate(VoxelHashTable hash_table, Vector3<short> *block_pos) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  hash_table.Allocate(block_pos[idx]);
}

__global__ void Retrieve(VoxelHashTable hash_table, 
                         const Vector3<short> *point, Voxel *voxel, VoxelBlock *voxel_block) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel[idx] = hash_table.Retrieve(point[idx], voxel_block[idx]);
}

__global__ void Assignment(VoxelHashTable hash_table, const Vector3<short> *point, Voxel *voxel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  VoxelBlock block;
  Voxel *voxel_old = hash_table.RetrieveMutable(point[idx], block);
  assert(voxel_old != NULL);
  *voxel_old = voxel[idx];
}


TEST_F(VoxelHashTest, Single) {
  // allocate block (1, 1, 1)
  *block_pos = Vector3<short>(1);
  *point = Vector3<short>(8);
  Allocate<<<1, 1>>>(voxel_hash_table, block_pos);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 1);
  EXPECT_EQ(voxel_block->block_pos, *block_pos);
  // retrieve empty block
  *point = Vector3<short>(0);
  Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  cudaDeviceSynchronize();
  EXPECT_EQ(voxel->weight, 0);
  // assignment
  *block_pos = Vector3<short>(0);
  Allocate<<<1, 1>>>(voxel_hash_table, block_pos);
  voxel_block->offset = 0; // reset cache after re allocation
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (unsigned char i = 0; i < BLOCK_LEN; ++i) {
    *point = { 0, 0, i };
    *voxel = { 1, { i, i, i }, i };
    Assignment<<<1, 1>>>(voxel_hash_table, point, voxel);
    cudaDeviceSynchronize();
  }
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 2);
  for (unsigned char i = 0; i < BLOCK_LEN; ++i) {
    *point = { 0, 0, i };
    Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaDeviceSynchronize();
    EXPECT_EQ(voxel->sdf, 1);
    EXPECT_EQ(voxel->rgb[0], i);
    EXPECT_EQ(voxel->rgb[1], i);
    EXPECT_EQ(voxel->rgb[2], i);
    EXPECT_EQ(voxel->weight, i);
  }
}

TEST_F(VoxelHashTest, Multiple) {
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    block_pos[i] = { i, i, i };
  }
  Allocate<<<1, MAX_BLOCKS>>>(voxel_hash_table, block_pos);
  voxel_hash_table.ResetLocks();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  // check received (assume no collision)
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), MAX_BLOCKS);
  // assign some voxels
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    point[i] = Vector3<short>(i * BLOCK_LEN);
    voxel[i] = { 1, { i, i, i }, i };
  }
  Assignment<<<1, MAX_BLOCKS>>>(voxel_hash_table, point, voxel);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  // reset buffer
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    voxel[i] = { 0, { 0, 0, 0 }, 0 };
    block_pos[i] = { 0, 0, 0 };
  }
  // retrieve and verify
  Retrieve<<<1, MAX_BLOCKS>>>(voxel_hash_table, point, voxel, voxel_block);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    EXPECT_EQ(voxel[i].sdf, 1);
    EXPECT_EQ(voxel[i].rgb[0], i);
    EXPECT_EQ(voxel[i].rgb[1], i);
    EXPECT_EQ(voxel[i].rgb[2], i);
    EXPECT_EQ(voxel[i].weight, i);
    EXPECT_EQ(voxel_block[i].block_pos, Vector3<short>(i));
  }
} 

TEST_F(VoxelHashTest, Collision) {
  // all hash to the very last index (1 << NUM_BUCKET_BITS) - 1
  block_pos[0] = { 33, 180, 42 };
  block_pos[1] = { 61, 16, 170 };
  block_pos[2] = { 63, 171, 45 };
  ASSERT_EQ(hash(block_pos[0]), hash(block_pos[1]));
  ASSERT_EQ(hash(block_pos[1]), hash(block_pos[2]));
  // hash to another idx
  block_pos[3] = { 0, 0, 0 };
  // allocate with conflict
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  voxel_hash_table.ResetLocks();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 2);
  // allocate again
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  voxel_hash_table.ResetLocks();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 3);
  // allocate yet again
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  voxel_hash_table.ResetLocks();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 4);
  // do some assignment
  for (unsigned char i = 0; i < 4; ++i) {
    point[i] = block_pos[i] * BLOCK_LEN; // use the first point of a block
    voxel[i] = { 1, { i, i, i }, i };
  }
  Assignment<<<1, 4>>>(voxel_hash_table, point, voxel);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  // reset buffer
  for (unsigned char i = 0; i < 4; ++i) {
    voxel[i] = { 0, { 0, 0, 0 }, 0 };
    block_pos[i] = { 0, 0, 0 };
  }
  // retrieve and verify
  Retrieve<<<1, 4>>>(voxel_hash_table, point, voxel, voxel_block);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (unsigned char i = 0; i < 4; ++i) {
    EXPECT_EQ(voxel[i].sdf, 1);
    EXPECT_EQ(voxel[i].rgb[0], i);
    EXPECT_EQ(voxel[i].rgb[1], i);
    EXPECT_EQ(voxel[i].rgb[2], i);
    EXPECT_EQ(voxel[i].weight, i);
  }
}
