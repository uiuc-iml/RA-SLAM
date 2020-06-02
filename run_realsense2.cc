#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "pangolin_viewer/viewer.h"

#include "openvslam/config.h"
#include "openvslam/system.h"

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <popl.hpp>
#include <spdlog/spdlog.h>

#define RS_WIDTH  640
#define RS_HEIGHT 480
#define RS_FPS    60

class SR300 {
 public:
  SR300() : align_to_color_(RS2_STREAM_COLOR) {
    cfg_.enable_stream(RS2_STREAM_DEPTH, 
        RS_WIDTH, RS_HEIGHT, rs2_format::RS2_FORMAT_Z16, RS_FPS);
    cfg_.enable_stream(RS2_STREAM_COLOR,
        RS_WIDTH, RS_HEIGHT, rs2_format::RS2_FORMAT_RGB8, RS_FPS);
    pipe_profile_ = pipe_.start(cfg_);
  }

  double get_depth_scale() const {
    auto sensor = pipe_profile_.get_device().first<rs2::depth_sensor>();
    return 1. / sensor.get_depth_scale();
  }

  rs2_intrinsics get_camera_intrinsics() const {
    auto color_stream = pipe_profile_.get_stream(RS2_STREAM_COLOR)
                                     .as<rs2::video_stream_profile>();
    return color_stream.get_intrinsics();
  }

  void get_rgbd_frame(cv::Mat *color_img, cv::Mat *depth_img) const {
    auto frameset = pipe_.wait_for_frames();
    frameset = align_to_color_.process(frameset);

    *color_img = cv::Mat(cv::Size(RS_WIDTH, RS_HEIGHT), CV_8UC3, 
        (void*)frameset.get_color_frame().get_data(), cv::Mat::AUTO_STEP);
    *depth_img = cv::Mat(cv::Size(RS_WIDTH, RS_HEIGHT), CV_16UC1,
        (void*)frameset.get_depth_frame().get_data(), cv::Mat::AUTO_STEP);
  }

 private:
  rs2::config cfg_;
  rs2::pipeline pipe_;
  rs2::pipeline_profile pipe_profile_;
  rs2::align align_to_color_;
};

void rgbd_tracking(const std::shared_ptr<openvslam::config> &cfg,
                   const std::string &vocab_file_path,
                   const SR300 &camera) {
  openvslam::system SLAM(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    cv::Mat color_img, depth_img;
    while (true) {
      camera.get_rgbd_frame(&color_img, &depth_img);
      const auto tp = std::chrono::steady_clock::now();
      const auto timestamp = 
        std::chrono::duration_cast<std::chrono::duration<double>>(tp - start).count();
      SLAM.feed_RGBD_frame(color_img, depth_img, timestamp);
      if (SLAM.terminate_is_requested())
        break;
    }
  });

  viewer.run();

  t.join();

  SLAM.shutdown();
}

std::shared_ptr<openvslam::config> get_config(const std::string &config_file_path,
                                              const SR300 &camera) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  // modify configuration based on realsense camera data
  // pre-defined stream profile
  yaml_node["Camera.fps"] = RS_FPS;
  yaml_node["Camera.cols"] = RS_WIDTH;
  yaml_node["Camera.rows"] = RS_HEIGHT;
  yaml_node["Camera.color_order"] = "RGB"; 
  // camera intrinsics
  rs2_intrinsics i = camera.get_camera_intrinsics();
  yaml_node["Camera.fx"] = i.fx;
  yaml_node["Camera.fy"] = i.fy;
  yaml_node["Camera.cx"] = i.ppx;
  yaml_node["Camera.cy"] = i.ppy;
  // zero camera distortion
  yaml_node["Camera.k1"] = 0;
  yaml_node["Camera.k2"] = 0;
  yaml_node["Camera.p1"] = 0;
  yaml_node["Camera.p2"] = 0;
  yaml_node["Camera.k3"] = 0;
  // depth factor
  yaml_node["depthmap_factor"] = camera.get_depth_scale();

  return std::make_shared<openvslam::config>(yaml_node, config_file_path);
}

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab",
                                                          "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config",
                                                           "config file path");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  try {
    op.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
    std::cerr << "Invalid Arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  if (debug_mode->is_set())
    spdlog::set_level(spdlog::level::debug);
  else
    spdlog::set_level(spdlog::level::info);

  SR300 camera;

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = get_config(config_file_path->value(), camera);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  rgbd_tracking(cfg, vocab_file_path->value(), camera);

  return EXIT_SUCCESS;
}
