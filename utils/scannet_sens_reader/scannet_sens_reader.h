#pragma once

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "third_party/scannet/sensorData.hpp"

using std::string;

class scannet_sens_reader {
 public:
  /**
   * @brief Constructor for the scannet .sens file reader. This class leverages
   * the example .sens decompression code provided by ScanNet official
   * at https://github.com/ScanNet/ScanNet/tree/master/SensReader/
   * 
   * Every instance of this class maintains manages exactly one .sens file.
   */
  scannet_sens_reader(const string & sens_filepath);

  /**
   * @brief Return camera intrinsic parameter fx/fy/cx/cy extracted from the sens file
   * 
   * @return: CameraIntrinsics
   */
  CameraIntrinsics<float> get_camera_intrinsics();

  /**
   * @brief Return camera extrinsic from depth camera to RGB camera
   * 
   * Note: it seems that scannet dataset is captured using a single RGBD camera
   * (which yields RGBD frames from a unified camera viewpoint), so this function
   * returns identity at this moment.
   * 
   * @return: SE3 identity transformation
   */
  SE3<float> get_camera_extrinsics();

  /**
   * @brief Return depth map factor for the underlying depth camera.
   * 
   * For every pixel in the depth map, (value / depth_factor) will yield its
   * depth in meters.
   * 
   * @return: depth map factor
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
   *  @brief Return the number of RGBD frames for the underlying stream
   * 
   *  @return size of the sensor stream
   */
  int get_size();

 private:
  ml::SensorData sd_;
};