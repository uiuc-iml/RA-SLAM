#pragma once

#include <assert.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
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
#include "utils/config_reader.hpp"
#include "utils/offline_data_provider/offline_data_provider.h"
#include "utils/rotation_math/pose_manager.h"

#include "modules/slam_module.h"

using std::string;

class rosbag_reader : public offline_data_provider {
 public:
  /**
   * @brief Constructor for the rosbag reader.
   *
   * Expected topics and types:
   *
   *  - depth_factor_topic  metadata topic containing string of depth factor
   *  - extrinsics_topic    metadata topic containing extrinsics from RGB to color camera
   *  - intrinsics_topic    metadata topic containing depth camera intrinsics
   *  - depth_img_topic     streaming topic containing streams of depth images
   *  - rgb_img_topic       streaming topic containing streams of rgb images
   *  - pose_topic          streaming topic containing streams of poses with timestamps
   */
  rosbag_reader(const string& rosbag_path);

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

  const std::string depth_factor_topic = "/camera/aligned_depth_to_color/depth_factor";
  const std::string extrinsics_topic = "/camera/aligned_depth_to_color/extrinsics";
  const std::string intrinsics_topic = "/camera/aligned_depth_to_color/intrinsics";
  const std::string depth_img_topic = "/camera/aligned_depth_to_color/image_raw";
  const std::string rgb_img_topic = "/camera/color/image_raw";
  const std::string pose_topic = "/zed2/zed_node/pose";
  const std::string left_img_topic = "/zed2/zed_node/left_raw/image_raw_color";
  const std::string right_img_topic = "/zed2/zed_node/right_raw/image_raw_color";
  const std::string camera_config_path = "/home/roger/disinfect-slam/configs/zed_SN28498913_l515.yaml";
  const std::string vocab_path = "/home/roger/Downloads/orb_vocab.dbow2";

  const static int width = 640;
  const static int height = 360;

 private:
  int size_;

  std::vector<cv::Mat> rgb_img_vec_;
  std::vector<cv::Mat> depth_img_vec_;
  std::vector<int64_t> rgb_ts_vec_;
  std::vector<int64_t> depth_ts_vec_;
  std::vector<SE3<float>> pose_vec_;

  std::string depth_factor_str_;
  std::string intrinsics_str_;
  std::string extrinsics_str_;

  /**
   * @brief Process camera metadata contained in ROSBag
   *
   * @param my_bag_ loaded ROSBag
   */
  void process_metadata(const rosbag::Bag &my_bag_);

  /**
   * @brief Process camera streaming data contained in ROSBag
   *
   * @param my_bag_ loaded ROSBag
   */
  void process_streamdata(const rosbag::Bag &my_bag_);

  /**
   * @brief Process stereo data from ZED camera and serialize them as camera poses
   * 
   * @param my_bag_ loaded ROSBag
   */
  void process_stereodata(const rosbag::Bag &my_bag_);
};