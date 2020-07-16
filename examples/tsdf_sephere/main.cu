#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_tsdf.cuh"

#include <iostream>
#include <opencv2/highgui.hpp>

#define LEN 8
#define Z_OFFSET 10 
#define X_CENTER (LEN >> 1)
#define Y_CENTER (LEN >> 1)
#define Z_CENTER (LEN >> 1)

#define GRID_Z_OFFSET (Z_OFFSET << BLOCK_LEN_BITS)

#define VOXEL_SIZE  (8./512)
#define TRUNCATION  0.04

__global__ void allocate_sephere_kernel(VoxelHashTable hash_table_) {
  hash_table_.Allocate({ 
    threadIdx.x - X_CENTER, 
    threadIdx.y - Y_CENTER, 
    threadIdx.z - Z_CENTER + Z_OFFSET,
  });
}

__global__ void assign_sephere_kernel(VoxelHashTable hash_table_) {
  __shared__ char buffer[sizeof(VoxelBlock)];
  VoxelBlock *block = (VoxelBlock*)buffer;
  const Vector3<short> block_pos(blockIdx.x - X_CENTER, 
                                 blockIdx.y - Y_CENTER, 
                                 blockIdx.z - Z_CENTER + Z_OFFSET);

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z) {
    hash_table_.Retrieve(block_pos << BLOCK_LEN_BITS, *block);  
  }
  __syncthreads();

  const Vector3<short> pos_grid = (block_pos << BLOCK_LEN_BITS) + 
                                  Vector3<short>(threadIdx.x, threadIdx.y, threadIdx.z);
  const Vector3<short> center_grid(0, 0, GRID_Z_OFFSET);
  const Vector3<float> vec_grid = (pos_grid - center_grid).cast<float>();
  const float vec_len = sqrtf(vec_grid.dot(vec_grid));

  Voxel &voxel = block->GetVoxelMutable(pos_grid);
  voxel.tsdf = vec_len - 20;
}

class TSDFGridTest : public TSDFGrid {
 public:
  using TSDFGrid::TSDFGrid;

  void AllocateSphere() {
    // call twice to ensure full allocation
    allocate_sephere_kernel<<<1, dim3(LEN, LEN, LEN)>>>(hash_table_);
    hash_table_.ResetLocks();
    allocate_sephere_kernel<<<1, dim3(LEN, LEN, LEN)>>>(hash_table_);
    hash_table_.ResetLocks();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    assert(hash_table_.NumActiveBlock() == LEN * LEN * LEN);
  }

  void AssignShpere() {
    const dim3 DIM(LEN, LEN, LEN);
    assign_sephere_kernel<<<DIM, DIM>>>(hash_table_);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};

int main() {
  TSDFGridTest tsdf(VOXEL_SIZE, TRUNCATION, 5);
  tsdf.AllocateSphere();
  tsdf.AssignShpere();
  // ray casting test
  const SE3<float> cam_P_world = SE3<float>::Identity();
  const CameraIntrinsics<float> intrinsics(350, 350, 320, 240);
  cv::Mat img_gray(480, 640, CV_32FC1);
  tsdf.RayCast(&img_gray, intrinsics, cam_P_world);

  cv::imshow("win", img_gray);
  cv::waitKey(0);
  return 0;
}
