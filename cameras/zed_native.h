#pragma once

#include <openvslam/camera/base.h>
#include <openvslam/config.h>

#include "utils/stereo_rectifier.h"
#include "utils/time.hpp"

#include <memory>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

class ZEDNative {
 public:
  ZEDNative(const openvslam::config &cfg, int dev_id = 0);
  ~ZEDNative();

  int64_t get_stereo_img(cv::Mat *left_img, cv::Mat* right_img) const ;

  void get_rectified_intrinsics(double *fx, double *fy, double *cx, double *cy, 
                                double *focal_x_baseline) const;

 private:
  void capture_thread();

  const openvslam::camera::base *cam_model_;
  const stereo_rectifier rectifier_;

  mutable cv::VideoCapture cap_;
};
