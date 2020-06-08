#pragma once

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

class ZED {
 public:
  ZED();

  ~ZED();

  sl::CameraConfiguration get_camera_config() const;

  void get_stereo_img(cv::Mat *left_img, cv::Mat *right_img);

 private:
  void allocate_if_needed(cv::Mat *img) const;

  sl::Camera zed_;
  sl::CameraConfiguration config_;
};
