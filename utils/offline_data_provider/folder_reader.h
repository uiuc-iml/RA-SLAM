#pragma once

#include <assert.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"
#include "utils/offline_data_provider/offline_data_provider.h"

using std::string;

struct LogEntry {
  int id;
  SE3<float> cam_T_world;
};

class folder_reader : public offline_data_provider {
 public:
  /**
   * @brief Constructor for the folder reader.
   *
   * Expected image folder format:
   *
   *  folder_path
   *    | id_depth.png
   *    | id_rgb.png
   *    | ...
   *    | trajectory.txt
   *    | camera_config.yaml
   *
   *  where id in images starts from 0 and can be arbitrarily large. Note each row in
   *  trajectory.txt only contains 12 row-major values (the last row 0 0 0 1 is omitted).
   */
  folder_reader(const string& folder_path);

  /**
   * @brief Return camera intrinsic parameter fx/fy/cx/cy extracted from the sens file
   *
   * @return CameraIntrinsics
   */
  CameraIntrinsics<float> get_camera_intrinsics();

  /**
   * @brief Return camera extrinsic from depth camera to RGB camera
   *
   * @return SE3 identity transformation
   */
  SE3<float> get_camera_extrinsics();

  /**
   * @brief Return depth map factor for the underlying depth camera.
   *
   * For every pixel in the depth map, (value / depth_factor) will yield its
   * depth in meters.
   *
   * @return depth map factor
   */
  float get_depth_map_factor();

  /**
   * @brief get depth frame by frame idx [0, this.get_size())
   *
   * @param depth_img a valid pointer to depth image to be written
   * @param frame_idx index of the wanted frame
   *
   */
  void get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx);

  /**
   * @brief get color frame by frame idx [0, this.get_size())
   *
   * @param depth_img a valid pointer to depth image to be written
   * @param frame_idx index of the wanted frame
   *
   */
  void get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx);

  /**
   * @brief get SE3 camera post by frame idx [0, this.get_size())
   *
   * @param frame_idx index of the wanted frame
   *
   * @return SE3 camera pose cam_T_world
   */
  SE3<float> get_camera_pose_by_id(int frame_idx);

  /**
   *  @brief Return the number of RGBD frames for the underlying stream
   *
   *  @return size of the sensor stream
   */
  int get_size();

  /**
   *  @brief Return the width of images in the sensor stream.
   *
   *  @return width of images
   */
  int get_width();

  /**
   *  @brief Return the height of images in the sensor stream.
   *
   *  @return height of images
   */
  int get_height();

 private:
  int width_;
  int height_;
  int size_;
  float depth_factor_;

  YAML::Node camera_config_;

  std::string logdir_;

  std::vector<LogEntry> log_entries_;

  /**
   * @brief Parse log entries
   *
   * @NOTE: this function originally resided in examples/tsdf/offline.cc
   *
   * @return a vector of logentry containing image id and corresponding camera poses
   */
  std::vector<LogEntry> parse_log_entries();
};
