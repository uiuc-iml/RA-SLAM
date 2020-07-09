#include "utils/tsdf/voxel_tsdf.cuh"

TSDFGrid::TSDFGrid(float voxel_size, float truncation, float max_depth) 
  : voxel_size_(voxel_size), truncation_(truncation), max_depth_(max_depth) {}

void TSDFGrid::RayCast(const Vector2<int> &img_hw, 
                       const CameraIntrinsics<float> &intrinsics, 
                       const SE3<float> &cam_T_world) {
  const Matrix3<float> intrinsics_inv = intrinsics.Inverse();
  const SE3<float> world_T_cam = cam_T_world.Inverse();
  const float step_len = truncation_ / 2;
  const float step_grid = step_len / voxel_size_;
  const int num_max_steps = static_cast<int>(max_depth_ / step_len);
}

