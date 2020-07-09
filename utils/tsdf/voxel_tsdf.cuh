#pragma once

#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/tsdf/voxel_hash.cuh"

class TSDFGrid {
 public:
  TSDFGrid(float voxel_size, float truncation, float max_depth);

  void RayCast(const Vector2<int> &img_hw, 
               const CameraIntrinsics<float> &intrinsics, 
               const SE3<float> &cam_T_world);

 protected:
  VoxelHashTable hash_table_;
  const float voxel_size_;
  const float truncation_;
  const float max_depth_;
};

