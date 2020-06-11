#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include <atomic>
#include <mutex>

typedef struct {
  sl::Mat left_img;
  sl::Mat right_img;
  sl::Mat xyzrgba;
} zed_meas_t;

class ZED {
 public:
  ZED();

  ~ZED();

  sl::CameraConfiguration get_camera_config() const;

  void get_stereo_img(cv::Mat *left_img, cv::Mat *right_img);

 private:
  void allocate_if_needed(cv::Mat *img) const;

  void capture_thread();

  void request_terminate();

  mutable std::mutex mtx_capture_;
  zed_meas_t measurements_[2];
  int write_idx_ = 0;

  mutable std::mutex mtx_terminate_;
  bool terminate_is_requested_ = false;
  std::unique_ptr<std::thread> capture_t_ = nullptr;

  sl::Camera zed_;
  sl::CameraConfiguration config_;
};
