#pragma once

#include <openvslam/publish/map_publisher.h>
#include <openvslam/system.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/stereo_rectifier.h"

inline CameraIntrinsics<float> GetIntrinsicsFromFile(const std::string& config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  return CameraIntrinsics<float>(config["Camera.fx"].as<float>(), config["Camera.fy"].as<float>(),
                                 config["Camera.cx"].as<float>(), config["Camera.cy"].as<float>());
}

inline int GetDepthFactorFromFile(const std::string& config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  return config["depthmap_factor"].as<float>();
}

inline SE3<float> GetExtrinsicsFromFile(const std::string& config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  const auto m = config["Extrinsics"].as<std::vector<double>>();
  const Eigen::Matrix4f tmp =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(m.data()).cast<float>();
  return SE3<float>(tmp);
}

inline std::shared_ptr<openvslam::config> GetAndSetConfig(const std::string& config_file_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  const StereoRectifier rectifier(yaml_node);
  const cv::Mat rectified_intrinsics = rectifier.RectifiedIntrinsics();
  yaml_node["Camera.fx"] = rectified_intrinsics.at<double>(0, 0);
  yaml_node["Camera.fy"] = rectified_intrinsics.at<double>(1, 1);
  yaml_node["Camera.cx"] = rectified_intrinsics.at<double>(0, 2);
  yaml_node["Camera.cy"] = rectified_intrinsics.at<double>(1, 2);
  yaml_node["Camera.focal_x_baseline"] = -rectified_intrinsics.at<double>(0, 3);
  return std::make_shared<openvslam::config>(yaml_node);
}
