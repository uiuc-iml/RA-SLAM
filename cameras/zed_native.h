#pragma once

#include <openvslam/camera/base.h>
#include <openvslam/config.h>

#include "utils/stereo_rectifier.h"
#include "utils/time.hpp"

#include <memory>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

/**
 * @brief ZED camera interface with OpenCV
 */
class ZEDNative {
 public:
  ZEDNative(const openvslam::config &cfg, int dev_id = 0);
  ~ZEDNative();

  /**
   * @brief get a stereo frame
   *
   * @param left_img  left image of stereo frame
   * @param right_img right image of stereo frame
   *
   * @return timestamp in system clock
   */
  int64_t GetStereoFrame(cv::Mat *left_img, cv::Mat* right_img) const ;

 private:
  const openvslam::camera::base *cam_model_;
  const StereoRectifier rectifier_;

  mutable cv::VideoCapture cap_;
};
