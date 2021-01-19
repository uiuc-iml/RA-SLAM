#pragma once

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

class L515 {
 public:
  L515();
  ~L515();

  double get_depth_scale() const;

  rs2_intrinsics get_camera_intrinsics() const;

  int64_t get_rgbd_frame(cv::Mat *color_img, cv::Mat *depth_img) const; 

  void set_depth_sensor_option(const rs2_option option, const float value);

  static const int WIDTH = 1280;
  static const int HEIGHT = 720;
  static const int FPS = 30;

 private:
  rs2::config cfg_;
  rs2::pipeline pipe_;
  rs2::pipeline_profile pipe_profile_;
  rs2::align align_to_color_;
};
