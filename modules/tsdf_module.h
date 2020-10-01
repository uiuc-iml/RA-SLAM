#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/gl/image.h"
#include "utils/tsdf/voxel_tsdf.cuh"

struct TSDFSystemInput {
  SE3<float> cam_P_world;
  cv::Mat img_rgb;
  cv::Mat img_depth;
  cv::Mat img_ht;
  cv::Mat img_lt;

  TSDFSystemInput(const SE3<float> &cam_P_world,
                  const cv::Mat &img_rgb, const cv::Mat &img_depth,
                  const cv::Mat &img_ht, const cv::Mat &img_lt)
    : cam_P_world(cam_P_world), img_rgb(img_rgb), img_depth(img_depth),
      img_ht(img_ht), img_lt(img_lt) {}
};

class TSDFSystem {
 public:
  TSDFSystem(float voxel_size, float truncation, float max_depth,
             const CameraIntrinsics<float> &intrinsics,
             const SE3<float> &extrinsics = SE3<float>::Identity());
  ~TSDFSystem();

  void Integrate(const SE3<float> &posecam_P_world,
                 const cv::Mat &img_rgb, const cv::Mat &img_depth,
                 const cv::Mat &img_ht = {}, const cv::Mat &img_lt = {});

  void Render(const CameraParams &virtual_cam,
              const SE3<float> cam_P_world,
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
  const SE3<float> cam_P_posecam_;
  // main integration thread
  std::thread t_;
  // input queue lock
  std::mutex mtx_queue_;
  std::queue<std::unique_ptr<TSDFSystemInput>> inputs_;
  // query lock
  std::mutex mtx_read_;
  // termination lock
  std::mutex mtx_terminate_;
  bool terminate_ = false;
};
