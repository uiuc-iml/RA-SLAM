#pragma once

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>
#include <string>

#include "utils/cuda/lie_group.cuh"
#include "utils/cuda/camera.cuh"
#include "utils/scannet_sens_reader/src/sensorData.hpp"

using std::string;

class scannet_sens_reader {
 public:
  /**
   * Constructor for the scannet .sens file reader. This class leverages
   * the example .sens decompression code provided by ScanNet official
   * at https://github.com/ScanNet/ScanNet/tree/master/SensReader/
   * 
   * Every instance of this class maintains manages exactly one .sens file.
   */
  scannet_sens_reader(const string & sens_filepath);

  /**
   * Return camera intrinsic parameter fx/fy/cx/cy extracted from the sens file
   * 
   * @return: CameraIntrinsics
   */
  CameraIntrinsics<float> get_camera_intrinsics();

  /**
   * Return camera extrinsic from depth camera to RGB camera
   * 
   * Note: it seems that scannet dataset is captured using a single RGBD camera
   * (which yields RGBD frames from a unified camera viewpoint), so this function
   * returns identity at this moment.
   * 
   * @return: SE3 identity transformation
   */
  SE3<float> get_camera_extrinsics();

  /**
   * Return depth map factor for the underlying depth camera.
   * 
   * For every pixel in the depth map, (value / depth_factor) will yield its
   * depth in meters.
   * 
   * @return: depth map factor
   */
  float get_depth_map_factor();

  /**
   *  Return the number of RGBD frames for the underlying stream
   */
  int get_size();

};