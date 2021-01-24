#include <iostream>
#include <string>
#include <thread>

#include <openvslam/system.h>
#include <openvslam/publish/map_publisher.h>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "modules/slam_module.h"
#include "modules/tsdf_module.h"
#include "utils/time.hpp"
#include "utils/gl/renderer_base.h"
#include "utils/cuda/errors.cuh"
#include "utils/rotation_math/pose_manager.h"
#include "utils/config_reader.hpp"
#include "modules/renderer_module.h"

void reconstruct(const ZEDNative &zed_native, const L515 &l515,
                 const std::shared_ptr<SLAMSystem> &SLAM,
                 const std::string &config_file_path) {
  // initialize TSDF
  auto TSDF = std::make_shared<TSDFSystem>(0.01, 0.06, 4,
      get_intrinsics_from_file(config_file_path), get_extrinsics_from_file(config_file_path));
  SLAM->startup();

  ImageRenderer renderer("tsdf", SLAM, TSDF, config_file_path);

  auto POSE_MANAGER = std::make_shared<pose_manager>();

  std::thread t_slam([&]() {
    while (true) {
      cv::Mat img_left, img_right;
      if (SLAM->terminate_is_requested())
        break;
      // get sensor readings
      const int64_t timestamp = zed_native.get_stereo_img(&img_left, &img_right);
      // visual slam
      const pose_valid_tuple m = SLAM->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
      const SE3<float> posecam_P_world(
        m.first(0, 0), m.first(0, 1), m.first(0, 2), m.first(0, 3),
        m.first(1, 0), m.first(1, 1), m.first(1, 2), m.first(1, 3),
        m.first(2, 0), m.first(2, 1), m.first(2, 2), m.first(2, 3),
        m.first(3, 0), m.first(3, 1), m.first(3, 2), m.first(3, 3)
      );
      if (m.second)
        POSE_MANAGER->register_valid_pose(timestamp, posecam_P_world);
    }
  });

  std::thread t_tsdf([&]() {
    const auto map_publisher = SLAM->get_map_publisher();
    while (true) {
      cv::Mat img_rgb, img_depth;
      if (SLAM->terminate_is_requested())
        break;
      const int64_t timestamp = l515.get_rgbd_frame(&img_rgb, &img_depth);
      const SE3<float> posecam_P_world = POSE_MANAGER->query_pose(timestamp);
      cv::resize(img_rgb, img_rgb, cv::Size(), .5, .5);
      cv::resize(img_depth, img_depth, cv::Size(), .5, .5);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / l515.get_depth_scale());
      TSDF->Integrate(posecam_P_world, img_rgb, img_depth);
    }
  });

  renderer.Run();
  t_slam.join();
  t_tsdf.join();
  SLAM->shutdown();
}

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config",
                                                           "config file path");
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
    cfg = get_and_set_config(config_file_path->value());
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  // initialize cameras
  ZEDNative zed_native(*cfg, device_id->value());
  L515 l515;
  // initialize slam
  auto SLAM = std::make_shared<SLAMSystem>(cfg, vocab_file_path->value());
  reconstruct(zed_native, l515, SLAM, config_file_path->value());

  return EXIT_SUCCESS;
}
