#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/gl/image.h"
#include "utils/tsdf/voxel_hash.cuh"

template <typename T>
using CubeVertices = Eigen::Matrix<T, 3, 1>[3];

typedef CubeVertices<float> CubeVerticesf;
typedef CubeVertices<double> CubeVerticesd;

template <typename T>
struct BoundingCube {
  T xmin;
  T xmax;
  T ymin;
  T ymax;
  T zmin;
  T zmax;

  template <typename Tout = T>
  BoundingCube<Tout> Scale(T scale) const {
    return BoundingCube<Tout>({static_cast<Tout>(xmin * scale), static_cast<Tout>(xmax * scale),
                               static_cast<Tout>(ymin * scale), static_cast<Tout>(ymax * scale),
                               static_cast<Tout>(zmin * scale), static_cast<Tout>(zmax * scale)});
  }
};

/**
 * @brief abstraction for TSDF grid
 */
class TSDFGrid {
 public:
  /**
   * @brief construct a TSDF grid
   *
   * @param voxel_size  size of a voxel in [m]
   * @param truncation  tsdf truncation in [m]
   */
  TSDFGrid(float voxel_size, float truncation);

  /**
   * @brief release GPU memory
   */
  ~TSDFGrid();

  /**
   * @brief integrate a single frame of data into the TSDF grid
   *
   * @param img_rgb     RGB image
   * @param img_depth   depth image in [m]
   * @param img_ht      high touch probability image
   * @param img_lt      low touch probability image
   * @param max_depth   maximum depth in [m]
   * @param intrinsics  camera intrinsics
   * @param cam_T_world transformation from camera to world
   */
  void Integrate(const cv::Mat& img_rgb, const cv::Mat& img_depth, const cv::Mat& img_ht,
                 const cv::Mat& img_lt, float max_depth, const CameraIntrinsics<float>& intrinsics,
                 const SE3<float>& cam_T_world);

  /**
   * @brief render a virtual view of the TSDF grid by ray casting
   *
   * @param max_depth   maximum depth in [m]
   * @param virtual_cam virtual camera parameterse (intrinsics + image sizes)
   * @param cam_T_world transformation from camera to world
   * @param tsdf_rgba   optional output image for rgb visualization
   * @param tsdf_normal optinoal output image for normal shaded visualization
   */
  void RayCast(float max_depth, const CameraParams& virtual_cam, const SE3<float>& cam_T_world,
               GLImage8UC4* tsdf_rgba = NULL, GLImage8UC4* tsdf_normal = NULL);

  /**
   * @brief gather all valid voxels
   *
   * @return an array of voxels with spatial location and tsdf values
   */
  std::vector<VoxelSpatialTSDF> GatherValid();

  /**
   * @brief gather all valid semantic voxels
   *
   * @return an array of voxels with spatial location and tsdf values
   */
  std::vector<VoxelSpatialTSDFSEGM> GatherValidSemantic();

  /**
   * @brief gather all voxels within certain bound
   *
   * @param volumn volumn of interest
   *
   * @return an array of voxels with spatial location and tsdf values
   */
  std::vector<VoxelSpatialTSDF> GatherVoxels(const BoundingCube<float>& volumn);

  void GatherValidMesh(std::vector<Eigen::Vector3f>* vertex_buffer,
                       std::vector<Eigen::Vector3i>* index_buffer);

 protected:
  void Allocate(const cv::Mat& img_rgb, const cv::Mat& img_depth, float max_depth,
                const CameraParams& cam_params, const SE3<float>& cam_T_world);

  int GatherVisible(float max_depth, const CameraParams& cam_params, const SE3<float>& cam_T_world);

  int GatherBlock();

  void UpdateTSDF(int num_visible_blocks, float max_depth, const CameraParams& cam_params,
                  const SE3<float>& cam_T_world);

  void SpaceCarving(int num_visible_blocks);

  cudaStream_t stream_;
  cudaStream_t stream2_;
  // voxel grid params
  VoxelHashTable hash_table_;
  const float voxel_size_;
  const float truncation_;

  // visibility buffer
  VoxelBlock* visible_blocks_;
  int* visible_mask_;
  int* visible_indics_;
  int* visible_indics_aux_;
  // image data buffer
  uchar3* img_rgb_;
  float* img_depth_;
  float* img_ht_;
  float* img_lt_;
  float* img_depth_to_range_;
  // renderer buffer
  uchar4* img_tsdf_rgba_;
  uchar4* img_tsdf_normal_;
};
