#include "zed_native.h"

#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

ZEDNative::ZEDNative(const openvslam::config &cfg, int dev_id) 
    : rectifier_(cfg.camera_, cfg.yaml_node_), cam_model_(cfg.camera_),
      cap_(dev_id),
      sgm_(cv::StereoSGBM::create(0, 128, 9, 4*3*81, 16*3*81, 1, 63, 10, 100, 32, cv::StereoSGBM::MODE_HH)){
  if (!cap_.isOpened()) {
    spdlog::critical("Cannot open zed camera on device {}", dev_id);
    exit(EXIT_FAILURE);
  }
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, cam_model_->cols_*2);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cam_model_->rows_);
  cap_.set(cv::CAP_PROP_FPS, cam_model_->fps_);
}

ZEDNative::~ZEDNative() {
  cap_.release();
}

void ZEDNative::get_stereo_img(cv::Mat *left_img, cv::Mat *right_img) {
  cv::Mat raw_img;
  if (cap_.read(raw_img)) {
    rectifier_.rectify(
        raw_img(cv::Rect(0, 0, cam_model_->cols_, cam_model_->rows_)),
        raw_img(cv::Rect(cam_model_->cols_, 0, cam_model_->cols_, cam_model_->rows_)),
        *left_img, *right_img);
  }
}

cv::Mat ZEDNative::compute_depth(const cv::Mat &left_img, const cv::Mat &right_img) {
  cv::Mat disparity;
  sgm_->compute(left_img, right_img, disparity);
  return disparity;
}
