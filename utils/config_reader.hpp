#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <openvslam/system.h>
#include <openvslam/publish/map_publisher.h>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/stereo_rectifier.h"

inline CameraIntrinsics<float> get_intrinsics_from_file(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  return CameraIntrinsics<float>(config["Camera.fx"].as<float>(),
                                 config["Camera.fy"].as<float>(),
                                 config["Camera.cx"].as<float>(),
                                 config["Camera.cy"].as<float>());
}

inline int get_depth_factor_from_file(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  return config["depthmap_factor"].as<float>();
}

inline SE3<float> get_extrinsics_from_file(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  const auto m = config["Extrinsics"].as<std::vector<double>>();
  return SE3<float>(
    m[0], m[1], m[2], m[3],
    m[4], m[5], m[6], m[7],
    m[8], m[9], m[10], m[11],
    m[12], m[13], m[14], m[15]
  );
}

inline std::shared_ptr<openvslam::config> get_and_set_config(const std::string &config_file_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  const stereo_rectifier rectifier(yaml_node);
  const cv::Mat rectified_intrinsics = rectifier.get_rectified_intrinsics();
  yaml_node["Camera.fx"] = rectified_intrinsics.at<double>(0, 0);
  yaml_node["Camera.fy"] = rectified_intrinsics.at<double>(1, 1);
  yaml_node["Camera.cx"] = rectified_intrinsics.at<double>(0, 2);
  yaml_node["Camera.cy"] = rectified_intrinsics.at<double>(1, 2);
  yaml_node["Camera.focal_x_baseline"] = -rectified_intrinsics.at<double>(0, 3);
  return std::make_shared<openvslam::config>(yaml_node);
}