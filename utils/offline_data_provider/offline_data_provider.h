#pragma once

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "third_party/scannet/sensorData.hpp"
#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"

using std::string;

class offline_data_provider {
 public:
  /**
   * @brief Return camera intrinsic parameter fx/fy/cx/cy extracted
   *
   * @return CameraIntrinsics
   */
  virtual CameraIntrinsics<float> get_camera_intrinsics() = 0;

  /**
   * @brief Return camera extrinsic from depth camera to RGB camera
   *
   * @return SE3 identity transformation
   */
  virtual SE3<float> get_camera_extrinsics() = 0;

  /**
   * @brief Return depth map factor for the underlying depth camera.
   *
   * For every pixel in the depth map, (value / depth_factor) will yield its
   * depth in meters.
   *
   * @return depth map factor
   */
  virtual float get_depth_map_factor() = 0;

  /**
   * @brief get depth frame by frame idx [0, this.get_size())
   *
   * @param depth_img a valid pointer to depth image to be written
   * @param frame_idx index of the wanted frame
   *
   */
  virtual void get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx) = 0;

  /**
   * @brief get color frame by frame idx [0, this.get_size())
   *
   * @param rgb_img a valid pointer to rgb image to be written. Channel must be RGB (not opencv
   * default BGR)
   * @param frame_idx index of the wanted frame
   *
   */
  virtual void get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx) = 0;

  /**
   * @brief get SE3 camera post by frame idx [0, this.get_size())
   *
   * @param frame_idx index of the wanted frame
   *
   * @return SE3 camera pose cam_T_world
   */
  virtual SE3<float> get_camera_pose_by_id(int frame_idx) = 0;

  /**
   *  @brief Return the number of RGBD frames for the underlying stream
   *
   *  @return size of the sensor stream
   */
  virtual int get_size() = 0;

  /**
   *  @brief Return the width of images in the sensor stream.
   *
   *  @return width of images
   */
  virtual int get_width() = 0;

  /**
   *  @brief Return the height of images in the sensor stream.
   *
   *  @return height of images
   */
  virtual int get_height() = 0;
};