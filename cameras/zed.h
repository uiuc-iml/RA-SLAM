#pragma once

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

/**
 * @brief ZED camera interface using ZED SDK
 */
class ZED {
 public:
  ZED();
  ~ZED();

  /**
   * @return camera config including image specs and calibration parameters
   */
  sl::CameraConfiguration GetConfig() const;

  /**
   * @brief read both stereo and RGBD frames
   *
   * @param left_img    left image of stereo frame
   * @param right_img   right image of stereo frame
   * @param rgb_img     rgb image of RGBD frame
   * @param depth_img   depth image of RGBD frame
   */
  void GetStereoAndRGBDFrame(cv::Mat* left_img, cv::Mat* right_img, cv::Mat* rgb_img,
                             cv::Mat* depth_img);

 private:
  void AllocateIfNeeded(cv::Mat* img, int type) const;

  sl::Camera zed_;
  sl::CameraConfiguration config_;
  sl::RuntimeParameters rt_params_;
};
