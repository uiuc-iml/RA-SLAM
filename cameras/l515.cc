#include "l515.h"

#include <spdlog/spdlog.h>

#include "utils/time.hpp"

const int L515::FPS;
const int L515::WIDTH;
const int L515::HEIGHT;
const int L515::DEPTH_WIDTH;
const int L515::DEPTH_HEIGHT;

L515::L515() : align_to_color_(RS2_STREAM_COLOR) {
  cfg_.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, rs2_format::RS2_FORMAT_Z16, FPS);
  cfg_.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, rs2_format::RS2_FORMAT_RGB8, FPS);
  pipe_profile_ = pipe_.start(cfg_);
}

L515::~L515() { pipe_.stop(); }

double L515::DepthScale() const {
  const auto sensor = pipe_profile_.get_device().first<rs2::depth_sensor>();
  return 1. / sensor.get_depth_scale();
}

int64_t L515::GetRGBDFrame(cv::Mat* color_img, cv::Mat* depth_img) const {
  // synchronous API. Stall until frames are ready
  auto frameset = pipe_.wait_for_frames();

  //
  frameset = align_to_color_.process(frameset);

  rs2::frame color_frame = frameset.get_color_frame();
  rs2::frame depth_frame = frameset.get_depth_frame();

  /*
   * Note that though set with DEPTH_WIDTH, DEPTH_HEIGHT,
   * depth image has size of (WIDTH, HEIGHT) due to
   * implementation in librealsense
   */
  *color_img =
      cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
  *depth_img =
      cv::Mat(cv::Size(WIDTH, HEIGHT), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

  // Timestamp of depth frame and color frame is not exactly the same
  // But depth frame is used in reconstruction. So we are returning this.
  // int64_t timestamp = (int64_t)(depth_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP));
  // TODO MAJOR frame_metadata is 0 for some reason
  int64_t timestamp = (int64_t) (GetSystemTimestamp<std::chrono::milliseconds>());
  std::cout << "CAMERA:    " << timestamp << std::endl;
  return timestamp;
}

void L515::SetDepthSensorOption(const rs2_option option, const float value) {
  auto sensor = pipe_profile_.get_device().first<rs2::depth_sensor>();
  if (!sensor.supports(option)) {
    spdlog::error("{} not supported", sensor.get_option_description(option));
    return;
  }
  const auto option_range = sensor.get_option_range(option);
  if (value < option_range.min || value > option_range.max) {
    spdlog::error("value {} out of range ([{}, {}])", value, option_range.min, option_range.max);
    return;
  }
  try {
    sensor.set_option(option, value);
  } catch (const rs2::error& e) {
    spdlog::error("Failed to set option: {}", e.what());
  }
}
