#pragma once

#include "openvslam/camera/base.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"

#include <memory>

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

class ZEDNative {
 public:
  ZEDNative(const openvslam::config &cfg);
  ~ZEDNative();

  void get_stereo_img(cv::Mat *left_img, cv::Mat* right_img);

  cv::Mat compute_depth(const cv::Mat &left_img, const cv::Mat &right_img);

 private:
  void capture_thread();

  const openvslam::camera::base *cam_model_;
  const openvslam::util::stereo_rectifier rectifier_;

  int dev_id_;
  std::unique_ptr<cv::VideoCapture> cap_ = nullptr;
  cv::Ptr<cv::StereoSGBM> sgm_;
};
