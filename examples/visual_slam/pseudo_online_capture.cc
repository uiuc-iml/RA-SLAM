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

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "modules/slam_module.h"
#include "utils/data_logger.hpp"
#include "utils/time.hpp"

class SemanticSLAMData {
 public:
  SemanticSLAMData() = default;

  SemanticSLAMData(const SemanticSLAMData &others)
    : zed_img_left(others.zed_img_left.clone()),
      zed_img_right(others.zed_img_right.clone()),
      l515_img_rgb(others.l515_img_rgb.clone()),
      l515_img_depth(others.l515_img_depth.clone()),
      id(others.id) {}

  cv::Mat zed_img_left;
  cv::Mat zed_img_right;
  cv::Mat l515_img_rgb;
  cv::Mat l515_img_depth;
  unsigned int id;
};

class SemanticSLAMLogger : public DataLogger<SemanticSLAMData> {
 public:
  SemanticSLAMLogger(const std::string &logdir)
      : logdir_(logdir),
        DataLogger<SemanticSLAMData>() {}

  std::vector<unsigned int> logged_ids;

 protected:
  void SaveData(const SemanticSLAMData &data) override {
    const std::string rgb_path = logdir_ + "/" + std::to_string(data.id) + "_rgb.png";
    const std::string depth_path = logdir_ + "/" + std::to_string(data.id) + "_depth.png";
    const std::string left_path = logdir_ + "/" + std::to_string(data.id) + "_left.png";
    const std::string right_path = logdir_ + "/" + std::to_string(data.id) + "_right.png";
    cv::Mat l515_depth_uint16;
    data.l515_img_depth.convertTo(l515_depth_uint16, CV_16UC1);
    // write!
    cv::imwrite(rgb_path, data.l515_img_rgb);
    cv::imwrite(depth_path, l515_depth_uint16);
    cv::imwrite(left_path, data.zed_img_left);
    cv::imwrite(right_path, data.zed_img_right);
    logged_ids.push_back(data.id);
  }

 private:
  const std::string logdir_;
};

void tracking(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path,
              const std::string &map_db_path,
              const std::string &logdir,
              const L515 &l515,
              const ZEDNative &zed_native) {
  SLAMSystem SLAM(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  SemanticSLAMLogger logger(logdir);

  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    SemanticSLAMData data;
    while (true) {
      if (SLAM.terminate_is_requested())
        break;

      zed_native.GetStereoFrame(&data.zed_img_left, &data.zed_img_right);
      l515.GetRGBDFrame(&data.l515_img_rgb, &data.l515_img_depth);
      const auto timestamp = GetTimestamp<std::chrono::microseconds>();

      data.id = SLAM.FeedStereoImages(
          data.zed_img_left, data.zed_img_right, timestamp / 1e6);

      logger.LogData(data);
    }
  });

  viewer.run();
  t.join();
  SLAM.shutdown();

  if (!map_db_path.empty())
    SLAM.save_map_database(map_db_path);

  const std::string traj_path = logdir + "/trajectory.txt";
  SLAM.SaveMatchedTrajectory(traj_path, logger.logged_ids);
}

std::shared_ptr<openvslam::config> get_and_set_config(const std::string &config_file_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  const StereoRectifier rectifier(yaml_node);
  const cv::Mat rectified_intrinsics = rectifier.RectifiedIntrinsics();
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

  ZEDNative zed_native(*cfg, device_id->value());
  L515 l515;

  tracking(cfg,
      vocab_file_path->value(), map_db_path->value(), log_dir->value(), l515, zed_native);

  return EXIT_SUCCESS;
}
