#include "zed.h"

#include <spdlog/spdlog.h>

ZED::ZED() {
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::VGA;
  init_params.camera_fps = 100;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.coordinate_units = sl::UNIT::METER;
  zed_.open(init_params);
  config_ = zed_.getCameraInformation().camera_configuration;
  spdlog::debug("CONSTRUCT: ZED::ZED");
}

ZED::~ZED() {
  zed_.close();
}

sl::CameraConfiguration ZED::get_camera_config() const {
  return config_;
}

void ZED::get_stereo_img(cv::Mat *left_img, cv::Mat *right_img) {
  allocate_if_needed(left_img);
  allocate_if_needed(right_img);
  
  sl::Mat left_sl(config_.resolution, sl::MAT_TYPE::U8_C1, 
                  left_img->data, config_.resolution.width);
  sl::Mat right_sl(config_.resolution, sl::MAT_TYPE::U8_C1, 
                   right_img->data, config_.resolution.width);

  if (zed_.grab() == sl::ERROR_CODE::SUCCESS) {
    zed_.retrieveImage(left_sl, sl::VIEW::LEFT_GRAY);
    zed_.retrieveImage(right_sl, sl::VIEW::RIGHT_GRAY);
  }
}

void ZED::allocate_if_needed(cv::Mat *img) const {
  if (img->empty() || img->type() != CV_8UC1 || 
      img->cols != config_.resolution.width ||
      img->rows != config_.resolution.height)
    *img = cv::Mat(config_.resolution.height, config_.resolution.width, CV_8UC1);
}
