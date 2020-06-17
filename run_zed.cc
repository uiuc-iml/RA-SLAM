#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "pangolin_viewer/viewer.h"

#include "openvslam/config.h"

#include <opencv2/opencv.hpp>

#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "cameras/zed_native.h"
#include "modules/slam_module.h"
#include "utils/time.hpp"

class StereoData {
 public:
  cv::Mat img_left;
  cv::Mat img_right;
  unsigned int id;
};

class StereoLogger {
 public:
  StereoLogger(const std::string &logdir) 
      : logdir_(logdir),
        log_thread_(&StereoLogger::run, this) {}

  ~StereoLogger() {
    {
      std::lock_guard<std::mutex> lock(mtx_terminate_);
      terminate_is_requested_ = true;
    }
    log_thread_.join();
  }

  void put_stereo_data(const cv::Mat &img_left, const cv::Mat &img_right, const unsigned int id) {
    std::lock_guard<std::mutex> lock(mtx_data_);
    data_[write_idx_].id = id;
    data_[write_idx_].img_left = img_left.clone();
    data_[write_idx_].img_right = img_right.clone();
    data_available_ = true;
  }

  std::vector<unsigned int> logged_ids;

 private:
  const std::string logdir_;

  std::mutex mtx_data_;
  int write_idx_ = 0;
  bool data_available_ = false;
  StereoData data_[2];

  std::thread log_thread_;
  mutable std::mutex mtx_terminate_;
  bool terminate_is_requested_ = false;

  void save_latest_stereo_data() {
    // switch read / write buffer
    {
      std::lock_guard<std::mutex> lock(mtx_data_);
      if (!data_available_)
        return;
      data_available_ = false;
      write_idx_ = 1 - write_idx_;
    }
    const auto &stereo_data = data_[1 - write_idx_];
    if (stereo_data.img_left.empty() || stereo_data.img_right.empty())
      return;

    const std::string left_path = logdir_ + "/" + std::to_string(stereo_data.id) + "_left.png";
    const std::string right_path = logdir_ + "/" + std::to_string(stereo_data.id) + "_right.png";
    cv::imwrite(left_path, stereo_data.img_left);
    cv::imwrite(right_path, stereo_data.img_right);
    logged_ids.push_back(stereo_data.id);
  }

  void run() {
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        if (terminate_is_requested_)
          break;
      }
      save_latest_stereo_data();
    }
  }
};

void tracking(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path,
              const std::string &map_db_path,
              const std::string &logdir,
              ZEDNative *camera) {
  slam_system SLAM(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  StereoLogger logger(logdir);

  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    cv::Mat left_img, right_img;
    while (true) {
      if (SLAM.terminate_is_requested())
        break;

      camera->get_stereo_img(&left_img, &right_img);
      const auto timestamp = get_timestamp<std::chrono::microseconds>();

      const unsigned int id = SLAM.feed_stereo_images(left_img, right_img, timestamp / 1e6);

      logger.put_stereo_data(left_img, right_img, id);
    }
  });

  viewer.run();
  t.join();
  SLAM.shutdown();

  if (!map_db_path.empty())
    SLAM.save_map_database(map_db_path);

  const std::string traj_path = logdir + "/trajectory.txt";
  SLAM.save_matched_trajectory(traj_path, logger.logged_ids);
}

std::shared_ptr<openvslam::config> get_and_set_config(const std::string &config_file_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  const stereo_rectifier rectifier(yaml_node);
  const cv::Mat rectified_intrinsics = rectifier.get_rectified_intrinsics();
  yaml_node["Camera.fx"] = rectified_intrinsics.at<double>(0, 0);
  yaml_node["Camera.fy"] = rectified_intrinsics.at<double>(1, 1);
  yaml_node["Camera.cx"] = rectified_intrinsics.at<double>(0, 2);
  yaml_node["Camera.cy"] = rectified_intrinsics.at<double>(1, 2);
  yaml_node["Camera.focal_x_baseline"] = -rectified_intrinsics.at<double>(0, 3);
  return std::make_shared<openvslam::config>(yaml_node);
}

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab",
                                                          "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config",
                                                           "config file path");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db",
                            "path to store the map database", "");
  auto log_dir = op.add<popl::Value<std::string>>("", "logdir", 
                            "directory to store logged data", "./log");
  auto device_id = op.add<popl::Value<int>>("", "devid", "camera device id", 0);

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

  YAML::Node yaml_node = YAML::LoadFile(config_file_path->value());

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = get_and_set_config(config_file_path->value());
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  ZEDNative camera(*cfg, device_id->value());

  tracking(cfg, 
      vocab_file_path->value(), map_db_path->value(), log_dir->value(), &camera);

  return EXIT_SUCCESS;
}
