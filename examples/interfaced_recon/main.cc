#include <iostream>
#include <string>
#include <thread>

#include <openvslam/system.h>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "utils/time.hpp"
#include "utils/config_reader.hpp"
#include "disinfect_slam/disinfect_slam.h"

void run(const ZEDNative &zed_native, const L515 &l515, std::shared_ptr<DISINFSystem> my_sys) {
  // initialize TSDF

  std::thread t_slam([&]() {
    while (true) {
      cv::Mat img_left, img_right;
      const int64_t timestamp = zed_native.GetStereoFrame(&img_left, &img_right);
      my_sys->feed_stereo_frame(img_left, img_right, timestamp);
    }
  });

  std::thread t_tsdf([&]() {
    while (true) {
      cv::Mat img_rgb, img_depth;
      const int64_t timestamp = l515.GetRGBDFrame(&img_rgb, &img_depth);
      my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp);
    }
  });

  my_sys->run();
  t_slam.join();
  t_tsdf.join();
}

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config",
                                                           "config file path");
  auto seg_model_path = op.add<popl::Value<std::string>>("m", "model",
                                                            "PyTorch JIT traced model path");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
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

  if (!vocab_file_path->is_set() || !config_file_path->is_set() ||!seg_model_path->is_set()) {
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
    cfg = GetAndSetConfig(config_file_path->value());
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  // initialize cameras
  ZEDNative zed_native(*cfg, device_id->value());
  L515 l515;
  // initialize slam
  std::shared_ptr<DISINFSystem> my_system = std::make_shared<DISINFSystem>(
      config_file_path->value(),
      vocab_file_path->value(),
      seg_model_path->value(),
      true
  );

  run(zed_native, l515, my_system);

  return EXIT_SUCCESS;
}
