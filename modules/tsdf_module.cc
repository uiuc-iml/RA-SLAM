#include "modules/tsdf_module.h"

#include <spdlog/spdlog.h>

TSDFSystem::TSDFSystem(float voxel_size, float truncation, float max_depth,
                       const CameraIntrinsics<float>& intrinsics, const SE3<float>& extrinsics)
    : tsdf_(voxel_size, truncation),
      max_depth_(max_depth),
      intrinsics_(intrinsics),
      cam_T_posecam_(extrinsics),
      t_(&TSDFSystem::Run, this) {
  spdlog::debug(
      "[TSDF System] Constructing with camera intrinsics: fx: {} fy: "
      "{} cx: {} cy:{}",
      intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);
}

TSDFSystem::~TSDFSystem() {
  {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_ = true;
  }
  t_.join();
}

void TSDFSystem::Integrate(const SE3<float>& posecam_T_world, const cv::Mat& img_rgb,
                           const cv::Mat& img_depth, const cv::Mat& img_ht, const cv::Mat& img_lt) {
  std::lock_guard<std::mutex> lock(mtx_queue_);
  if (img_ht.empty() || img_lt.empty()) {
    inputs_.push(std::make_unique<TSDFSystemInput>(
        cam_T_posecam_ * posecam_T_world, img_rgb, img_depth,
        cv::Mat::ones(img_depth.rows, img_depth.cols, img_depth.type()),
        cv::Mat::ones(img_depth.rows, img_depth.cols, img_depth.type())));
  } else {
    inputs_.push(std::make_unique<TSDFSystemInput>(cam_T_posecam_ * posecam_T_world, img_rgb,
                                                   img_depth, img_ht, img_lt));
  }
}

std::vector<VoxelSpatialTSDF> TSDFSystem::Query(const BoundingCube<float>& volumn) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  return tsdf_.GatherVoxels(volumn);
}

void TSDFSystem::Render(const CameraParams& virtual_cam, const SE3<float> cam_T_world,
                        GLImage8UC4* img_normal) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  tsdf_.RayCast(max_depth_, virtual_cam, cam_T_world, nullptr, img_normal);
}

void TSDFSystem::Run() {
  while (true) {
    // check for termination
    {
      std::lock_guard<std::mutex> lock(mtx_terminate_);
      if (terminate_) return;
    }
    // pop from input queue
    std::unique_ptr<TSDFSystemInput> input;
    {
      std::lock_guard<std::mutex> lock(mtx_queue_);
      if (inputs_.size() > 10)
        spdlog::warn("[TSDF System] Processing cannot catch up (input size: {})", inputs_.size());
      if (inputs_.empty()) continue;
      input = std::move(inputs_.front());
      inputs_.pop();
    }
    // tsdf integration
    {
      std::lock_guard<std::mutex> lock(mtx_read_);
      tsdf_.Integrate(input->img_rgb, input->img_depth, input->img_ht, input->img_lt, max_depth_,
                      intrinsics_, input->cam_T_world);
    }
  }
}
