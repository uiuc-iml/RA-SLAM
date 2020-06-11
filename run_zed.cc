#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "pangolin_viewer/viewer.h"

#include "openvslam/config.h"

#include <opencv2/opencv.hpp>

#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "cameras/zed_native.h"
#include "modules/slam_module.h"
#include "utils/time.hpp"

void tracking(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path,
              const std::string &map_db_path,
              ZEDNative *camera) {
  auto SLAM = openvslam::system(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer = pangolin_viewer::viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    cv::Mat left_img, right_img;
    while (true) {
      if (SLAM.terminate_is_requested())
        break;

      camera->get_stereo_img(&left_img, &right_img);
      const auto timestamp = get_timestamp<std::chrono::microseconds>();

      SLAM.feed_stereo_frame(left_img, right_img, timestamp / 1e6);
    }
  });

  viewer.run();
  t.join();
  SLAM.shutdown();

  if (!map_db_path.empty())
    SLAM.save_map_database(map_db_path);
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


  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = std::make_shared<openvslam::config>(config_file_path->value());
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  ZEDNative camera(*cfg, device_id->value());

  tracking(cfg, 
      vocab_file_path->value(), map_db_path->value(), &camera);

  return EXIT_SUCCESS;
}
