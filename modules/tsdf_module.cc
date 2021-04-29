#include "modules/tsdf_module.h"

#include <spdlog/spdlog.h>

#include "utils/time.hpp"

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

TSDFSystem::~TSDFSystem() { this->terminate(); }

void TSDFSystem::Integrate(const SE3<float>& posecam_T_world, const cv::Mat& img_rgb,
                           const cv::Mat& img_depth, const cv::Mat& img_ht, const cv::Mat& img_lt) {
  std::unique_lock<std::mutex> pause_lock(mtx_pause_);
  while (pause_) cv_pause_.wait(pause_lock);
  std::lock_guard<std::mutex> lock(mtx_queue_);
  if (img_ht.empty() || img_lt.empty()) {
    inputs_.push(std::make_unique<TSDFSystemInput>(
        cam_T_posecam_ * posecam_T_world, img_rgb.clone(), img_depth.clone(),
        cv::Mat::ones(img_depth.rows, img_depth.cols, img_depth.type()),
        cv::Mat::ones(img_depth.rows, img_depth.cols, img_depth.type())));
  } else {
    inputs_.push(std::make_unique<TSDFSystemInput>(cam_T_posecam_ * posecam_T_world,
                                                   img_rgb.clone(), img_depth.clone(),
                                                   img_ht.clone(), img_lt.clone()));
  }
}

std::vector<VoxelSpatialTSDF> TSDFSystem::Query(const BoundingCube<float>& volumn) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  std::vector<VoxelSpatialTSDF> ret = tsdf_.GatherVoxels(volumn);
  return ret;
}

void TSDFSystem::Render(const CameraParams& virtual_cam, const SE3<float> cam_T_world,
                        GLImage8UC4* img_rgba, GLImage8UC4* img_normal) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  tsdf_.RayCast(max_depth_ * 2, virtual_cam, cam_T_world, img_rgba, img_normal);
}

void TSDFSystem::Render(const CameraParams& virtual_cam, const SE3<float> cam_T_world,
                        GLImage8UC4* img_rgba, GLImage8UC4* img_normal, float max_depth) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  tsdf_.RayCast(max_depth, virtual_cam, cam_T_world, img_rgba, img_normal);
}

void TSDFSystem::DownloadAll(const std::string& file_path) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  const auto voxel_pos_prob = tsdf_.GatherValidSemantic();
  spdlog::debug("Visible TSDF blocks count: {}", voxel_pos_prob.size());
  std::ofstream fout(file_path, std::ios::out | std::ios::binary);
  fout.write((char*)voxel_pos_prob.data(), voxel_pos_prob.size() * sizeof(VoxelSpatialTSDFSEGM));
  fout.close();
}

void TSDFSystem::DownloadAllMesh(const std::string& vertices_path, const std::string& indices_path) {
  std::lock_guard<std::mutex> lock(mtx_read_);
  std::vector<Eigen::Vector3f> vertex_buffer;
  std::vector<Eigen::Vector3i> index_buffer;
  tsdf_.GatherValidMesh(&vertex_buffer, &index_buffer);
  spdlog::debug("Dumped triangle vertices count: {}", vertex_buffer.size());
  spdlog::debug("Dumped indices count: {}", index_buffer.size());
  std::ofstream vout(vertices_path, std::ios::out | std::ios::binary);
  std::ofstream iout(indices_path, std::ios::out | std::ios::binary);

  vout.write((char*)vertex_buffer.data(), vertex_buffer.size() * sizeof(Eigen::Vector3f));
  iout.write((char*)index_buffer.data(), index_buffer.size() * sizeof(Eigen::Vector3i));

  vout.close();
  iout.close();
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
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.Integrate(input->img_rgb, input->img_depth, input->img_ht, input->img_lt, max_depth_,
                      intrinsics_, input->cam_T_world);
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("[TSDF Module] Integration took: {} ms", end - st);
    }
  }
}

bool TSDFSystem::is_terminated() { return terminate_; }

void TSDFSystem::terminate() {
  {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_ = true;
  }
  t_.join();
}

void TSDFSystem::SetPause(bool pause) {
  std::unique_lock<std::mutex> pause_lock(mtx_pause_);
  pause_ = pause;
  cv_pause_.notify_all();
}
