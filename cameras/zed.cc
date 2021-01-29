#include "zed.h"

ZED::ZED() {
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::VGA;
  init_params.camera_fps = 30;
  init_params.coordinate_units = sl::UNIT::MILLIMETER;
  init_params.depth_mode = sl::DEPTH_MODE::QUALITY;
  zed_.open(init_params);
  zed_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, 100);
  rt_params_ = zed_.getRuntimeParameters();
  rt_params_.confidence_threshold = 50;
  config_ = zed_.getCameraInformation().camera_configuration;
}

ZED::~ZED() { zed_.close(); }

sl::CameraConfiguration ZED::GetConfig() const { return config_; }

void ZED::GetStereoAndRGBDFrame(cv::Mat* left_img, cv::Mat* right_img, cv::Mat* rgb_img,
                                cv::Mat* depth_img) {
  AllocateIfNeeded(left_img, CV_8UC1);
  AllocateIfNeeded(right_img, CV_8UC1);
  AllocateIfNeeded(rgb_img, CV_8UC4);
  AllocateIfNeeded(depth_img, CV_32FC1);

  sl::Mat left_sl(config_.resolution, sl::MAT_TYPE::U8_C1, left_img->data,
                  config_.resolution.width);
  sl::Mat right_sl(config_.resolution, sl::MAT_TYPE::U8_C1, right_img->data,
                   config_.resolution.width);
  sl::Mat rgb_sl(config_.resolution, sl::MAT_TYPE::U8_C4, rgb_img->data,
                 config_.resolution.width * 4);
  sl::Mat depth_sl(config_.resolution, sl::MAT_TYPE::F32_C1, depth_img->data,
                   config_.resolution.width * sizeof(float));

  if (zed_.grab(rt_params_) == sl::ERROR_CODE::SUCCESS) {
    zed_.retrieveImage(left_sl, sl::VIEW::LEFT_GRAY);
    zed_.retrieveImage(right_sl, sl::VIEW::RIGHT_GRAY);
    zed_.retrieveImage(rgb_sl, sl::VIEW::LEFT);
    zed_.retrieveMeasure(depth_sl, sl::MEASURE::DEPTH);
  }
}

void ZED::AllocateIfNeeded(cv::Mat* img, int type) const {
  if (img->empty() || img->type() != type || img->cols != config_.resolution.width ||
      img->rows != config_.resolution.height)
    *img = cv::Mat(config_.resolution.height, config_.resolution.width, type);
}
