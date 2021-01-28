#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/gl/image.h"
#include "utils/tsdf/voxel_tsdf.cuh"

/**
 * @brief input data structure for TSDF integration
 */
struct TSDFSystemInput {
  SE3<float> cam_T_world;
  cv::Mat img_rgb;
  cv::Mat img_depth;
  cv::Mat img_ht;
  cv::Mat img_lt;

  TSDFSystemInput(const SE3<float> &cam_T_world,
                  const cv::Mat &img_rgb, const cv::Mat &img_depth,
                  const cv::Mat &img_ht, const cv::Mat &img_lt)
    : cam_T_world(cam_T_world), img_rgb(img_rgb), img_depth(img_depth),
      img_ht(img_ht), img_lt(img_lt) {}
};

/**
 * @brief top level TSDF integration sytsem
 */
class TSDFSystem {
 public:
  /**
   * @brief constructor
   *
   * @param voxel_size  voxel resolution in [m]
   * @param truncation  tsdf truncation length in [m]
   * @param max_depth   maximum depth to be considered valid in [m]
   * @param intrinsics  intrinsics of the RGDB camera
   * @param extrinsics  optional extrinsics to a reference frame (transformation from
   *                    camera to reference frame: cam_T_ref)
   */
  TSDFSystem(float voxel_size, float truncation, float max_depth,
             const CameraIntrinsics<float> &intrinsics,
             const SE3<float> &extrinsics = SE3<float>::Identity());

  /**
   * @brief stop the integration thread
   */
  ~TSDFSystem();

  /**
   * @brief integrate a single frame of data into the TSDF grid
   *
   * @param posecam_T_world pose of the reference frame (i.e. pose camera)
   * @param img_rgb         RGB image
   * @param img_depth       depth image
   * @param img_ht          optional high touch probability image
   * @param img_lt          optional low touch probability image
   */
  void Integrate(const SE3<float> &posecam_T_world,
                 const cv::Mat &img_rgb, const cv::Mat &img_depth,
                 const cv::Mat &img_ht = {}, const cv::Mat &img_lt = {});

  /**
   * @brief download valid voxels within a certain bound
   *
   * @param volumn volumn of interests
   *
   * @return array of voxel data with spatial location and tsdf value
   */
  std::vector<VoxelSpatialTSDF> Query(const BoundingCube<float> &volumn);

  /**
   * @brief render a virtual view of the TSDF scene into a GLImage
   *
   * @param virtual_cam virtual camera paramteres
   * @param cam_T_world virtual camera pose
   * @param img_normal  output normal shaded image
   */
  void Render(const CameraParams &virtual_cam,
              const SE3<float> cam_T_world,
              GLImage8UC4 *img_normal);

 private:
  void Run();
  // TSDF grid
  TSDFGrid tsdf_;
  // maximum depth in TSDF integration
  float max_depth_;
  // instrsinsics of (undistorted) camera used for TSDF update
  const CameraIntrinsics<float> intrinsics_;
  // extrinsics w.r.t. pose camera
  const SE3<float> cam_T_posecam_;
  // input queue lock
  std::mutex mtx_queue_;
  std::queue<std::unique_ptr<TSDFSystemInput>> inputs_;
  // query lock
  std::mutex mtx_read_;
  // termination lock
  std::mutex mtx_terminate_;
  bool terminate_ = false;
  // main integration thread
  std::thread t_;
};
