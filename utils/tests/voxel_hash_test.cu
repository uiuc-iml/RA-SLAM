#include <gtest/gtest.h>

#include <cassert>

#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_hash.cuh"

#define MAX_BLOCKS 128

class VoxelHashTest : public ::testing::Test {
 protected:
  VoxelHashTest() {
    CUDA_SAFE_CALL(cudaMallocManaged(&voxel, sizeof(VoxelRGBW) * MAX_BLOCKS * BLOCK_VOLUME));
    CUDA_SAFE_CALL(cudaMallocManaged(&voxel_block, sizeof(VoxelBlock) * MAX_BLOCKS));
    CUDA_SAFE_CALL(cudaMallocManaged(&point, sizeof(Vector3<short>) * MAX_BLOCKS * BLOCK_VOLUME));
    CUDA_SAFE_CALL(cudaMallocManaged(&block_pos, sizeof(Vector3<short>) * MAX_BLOCKS));
  }

  ~VoxelHashTest() {
    voxel_hash_table.ReleaseMemory();
    CUDA_SAFE_CALL(cudaFree(voxel));
    CUDA_SAFE_CALL(cudaFree(voxel_block));
    CUDA_SAFE_CALL(cudaFree(point));
    CUDA_SAFE_CALL(cudaFree(block_pos));
  }

  VoxelHashTable voxel_hash_table;
  VoxelRGBW* voxel;
  VoxelBlock* voxel_block;
  Vector3<short>* point;
  Vector3<short>* block_pos;
};

__global__ void Allocate(VoxelHashTable hash_table, Vector3<short>* block_pos) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  hash_table.Allocate(block_pos[idx]);
}

__global__ void Retrieve(VoxelHashTable hash_table, const Vector3<short>* point, VoxelRGBW* voxel,
                         VoxelBlock* voxel_block) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  voxel[idx] = hash_table.Retrieve<VoxelRGBW>(point[idx], voxel_block[idx]);
}

__global__ void Assignment(VoxelHashTable hash_table, const Vector3<short>* point,
                           VoxelRGBW* voxel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  VoxelBlock block;
  VoxelRGBW* voxel_old = hash_table.RetrieveMutable<VoxelRGBW>(point[idx], block);
  assert(voxel_old != NULL);
  *voxel_old = voxel[idx];
}

TEST_F(VoxelHashTest, Single) {
  // allocate block (1, 1, 1)
  *block_pos = Vector3<short>(1);
  *point = Vector3<short>(8);
  Allocate<<<1, 1>>>(voxel_hash_table, block_pos);
  CUDA_CHECK_ERROR;
  Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 1);
  EXPECT_EQ(voxel_block->position, *block_pos);
  // retrieve empty block
  *point = Vector3<short>(0);
  Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_EQ(voxel->weight, 0);
  // assignment
  *block_pos = Vector3<short>(0);
  Allocate<<<1, 1>>>(voxel_hash_table, block_pos);
  CUDA_CHECK_ERROR;
  voxel_block->offset = 0;  // reset cache after re allocation
  for (unsigned char i = 0; i < BLOCK_LEN; ++i) {
    *point = {0, 0, i};
    *voxel = {{i, i, i}, i};
    Assignment<<<1, 1>>>(voxel_hash_table, point, voxel);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 2);
  for (unsigned char i = 0; i < BLOCK_LEN; ++i) {
    *point = {0, 0, i};
    Retrieve<<<1, 1>>>(voxel_hash_table, point, voxel, voxel_block);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    EXPECT_EQ(voxel->rgb.x, i);
    EXPECT_EQ(voxel->rgb.y, i);
    EXPECT_EQ(voxel->rgb.z, i);
    EXPECT_EQ(voxel->weight, i);
  }
}

TEST_F(VoxelHashTest, Multiple) {
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    block_pos[i] = {i, i, i};
  }
  Allocate<<<1, MAX_BLOCKS>>>(voxel_hash_table, block_pos);
  voxel_hash_table.ResetLocks();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  // check received (assume no collision)
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), MAX_BLOCKS);
  // assign some voxels
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    point[i] = Vector3<short>(i * BLOCK_LEN);
    voxel[i] = {{i, i, i}, i};
  }
  Assignment<<<1, MAX_BLOCKS>>>(voxel_hash_table, point, voxel);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  // reset buffer
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    voxel[i] = {{0, 0, 0}, 0};
    block_pos[i] = {0, 0, 0};
  }
  // retrieve and verify
  Retrieve<<<1, MAX_BLOCKS>>>(voxel_hash_table, point, voxel, voxel_block);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (unsigned char i = 0; i < MAX_BLOCKS; ++i) {
    EXPECT_EQ(voxel[i].rgb.x, i);
    EXPECT_EQ(voxel[i].rgb.y, i);
    EXPECT_EQ(voxel[i].rgb.z, i);
    EXPECT_EQ(voxel[i].weight, i);
    EXPECT_EQ(voxel_block[i].position, Vector3<short>(i));
  }
}

TEST_F(VoxelHashTest, Collision) {
  // all hash to the last index NUM_BUCKET - 1
  block_pos[0] = {33, 180, 42};
  block_pos[1] = {61, 16, 170};
  block_pos[2] = {63, 171, 45};
  ASSERT_EQ(Hash(block_pos[0]), NUM_BUCKET - 1);
  ASSERT_EQ(Hash(block_pos[0]), Hash(block_pos[1]));
  ASSERT_EQ(Hash(block_pos[1]), Hash(block_pos[2]));
  // hash to another idx
  block_pos[3] = {0, 0, 0};
  // allocate with conflict
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  CUDA_CHECK_ERROR;
  voxel_hash_table.ResetLocks();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 2);
  // allocate again
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  CUDA_CHECK_ERROR;
  voxel_hash_table.ResetLocks();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 3);
  // allocate yet again
  Allocate<<<1, 4>>>(voxel_hash_table, block_pos);
  CUDA_CHECK_ERROR;
  voxel_hash_table.ResetLocks();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  EXPECT_EQ(voxel_hash_table.NumActiveBlock(), 4);
  // do some assignment
  for (unsigned char i = 0; i < 4; ++i) {
    point[i] = block_pos[i] * BLOCK_LEN;  // use the first point of a block
    voxel[i] = {{i, i, i}, i};
  }
  Assignment<<<1, 4>>>(voxel_hash_table, point, voxel);
  CUDA_CHECK_ERROR;
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  // reset buffer
  for (unsigned char i = 0; i < 4; ++i) {
    voxel[i] = {{0, 0, 0}, 0};
    block_pos[i] = {0, 0, 0};
    voxel_block[i].idx = -1;
  }
  // retrieve and verify
  Retrieve<<<1, 4>>>(voxel_hash_table, point, voxel, voxel_block);
  CUDA_CHECK_ERROR;
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (unsigned char i = 0; i < 4; ++i) {
    EXPECT_EQ(voxel[i].rgb.x, i);
    EXPECT_EQ(voxel[i].rgb.y, i);
    EXPECT_EQ(voxel[i].rgb.z, i);
    EXPECT_EQ(voxel[i].weight, i);
  }
}
