#include "disinfect_slam/disinfect_slam.h"

#include "utils/time.hpp"

DISINFSystem::DISINFSystem(std::string camera_config_path, std::string vocab_path,
                           std::string seg_model_path, bool rendering_flag) {
  YAML::Node yaml_node = YAML::LoadFile(camera_config_path);
  tsdf_width_ = yaml_node["tsdf.width"].as<int>();
  tsdf_height_ = yaml_node["tsdf.height"].as<int>();
  std::shared_ptr<openvslam::config> cfg = GetAndSetConfig(camera_config_path);
  SLAM_ = std::make_shared<SLAMSystem>(cfg, vocab_path);
  SEG_ = std::make_shared<inference_engine>(seg_model_path, tsdf_width_, tsdf_height_);
  TSDF_ = std::make_shared<TSDFSystem>(0.01, 0.06, 4, GetIntrinsicsFromFile(camera_config_path),
                                       GetExtrinsicsFromFile(camera_config_path));

  depthmap_factor_ = GetDepthFactorFromFile(camera_config_path);

  SLAM_->startup();

  camera_pose_manager_ = std::make_shared<pose_manager>();

  if (rendering_flag) {
    RENDERER_ = std::make_shared<ImageRenderer>(
        "tsdf", std::bind(&pose_manager::get_latest_pose, camera_pose_manager_), TSDF_,
        GetIntrinsicsFromFile(camera_config_path));
  }
}

DISINFSystem::~DISINFSystem() { SLAM_->shutdown(); }

void DISINFSystem::run() { RENDERER_->Run(); }

void DISINFSystem::feed_rgbd_frame(const cv::Mat& img_rgb, const cv::Mat& img_depth,
                                   int64_t timestamp) {
  cv::Mat my_img_rgb, my_img_depth;  // local Mat that will be modified
  const SE3<float> posecam_P_world = camera_pose_manager_->query_pose(timestamp);
  cv::resize(img_rgb, my_img_rgb, cv::Size(tsdf_width_, tsdf_height_));
  cv::resize(img_depth, my_img_depth, cv::Size(tsdf_width_, tsdf_height_));
  my_img_depth.convertTo(my_img_depth, CV_32FC1,
                         1. / depthmap_factor_);  // depth scale
  std::vector<cv::Mat> prob_map = SEG_->infer_one(my_img_rgb, false);
  TSDF_->Integrate(posecam_P_world, my_img_rgb, my_img_depth, prob_map[0], prob_map[1]);
}

void DISINFSystem::feed_stereo_frame(const cv::Mat& img_left, const cv::Mat& img_right,
                                     int64_t timestamp) {
  const pose_valid_tuple m =
      SLAM_->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
  const SE3<float> posecam_P_world(m.first.cast<float>().eval());
  if (m.second) camera_pose_manager_->register_valid_pose(timestamp, posecam_P_world);
}

SE3<float> DISINFSystem::query_camera_pose(const int64_t timestamp) {
  return this->camera_pose_manager_->query_pose(timestamp);
}

std::vector<VoxelSpatialTSDF> DISINFSystem::query_tsdf(const BoundingCube<float>& volumn) {
  return TSDF_->Query(volumn);
}
