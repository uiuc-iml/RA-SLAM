#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/gl/image.h"
#include "utils/tsdf/voxel_hash.cuh"

class TSDFGrid {
 public:
  TSDFGrid(float voxel_size, float truncation);
  ~TSDFGrid();

  void Integrate(const cv::Mat &img_rgb, const cv::Mat &img_depth,
                 const cv::Mat &img_ht, const cv::Mat &img_lt,
                 float max_depth,
                 const CameraIntrinsics<float> &intrinsics,
                 const SE3<float> &cam_P_world);

  void RayCast(float max_depth,
               const CameraParams &virtual_cam,
               const SE3<float> &cam_P_world,
               GLImage8UC4 *tsdf_rgba = NULL, GLImage8UC4 *tsdf_normal = NULL);

  std::vector<Vector4<float>> GatherValid();

 protected:
  void Allocate(const cv::Mat &img_rgb, const cv::Mat &img_depth, float max_depth,
                const CameraParams &cam_params, const SE3<float> &cam_P_world);

  int GatherVisible(float max_depth,
                    const CameraParams &cam_params, const SE3<float> &cam_P_world);

  int GatherBlock();

  void UpdateTSDF(int num_visible_blocks, float max_depth,
                  const CameraParams &cam_params, const SE3<float> &cam_P_world);

  void SpaceCarving(int num_visible_blocks);

  cudaStream_t stream_;
  cudaStream_t stream2_;
  // voxel grid params
  VoxelHashTable hash_table_;
  const float voxel_size_;
  const float truncation_;

  // visibility buffer
  VoxelBlock *visible_blocks_;
  int *visible_mask_;
  int *visible_indics_;
  int *visible_indics_aux_;
  // image data buffer
  uchar3 *img_rgb_;
  float *img_depth_;
  float *img_ht_;
  float *img_lt_;
  float *img_depth_to_range_;
  // renderer buffer
  uchar4 *img_tsdf_rgba_;
  uchar4 *img_tsdf_normal_;
};

