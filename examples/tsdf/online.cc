#include <openvslam/publish/map_publisher.h>
#include <openvslam/system.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <popl.hpp>
#include <string>
#include <thread>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "modules/renderer_module.h"
#include "modules/slam_module.h"
#include "modules/tsdf_module.h"
#include "segmentation/inference.h"
#include "utils/config_reader.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/gl/renderer_base.h"
#include "utils/rotation_math/pose_manager.h"
#include "utils/time.hpp"

void reconstruct(const ZEDNative& zed_native, const L515& l515,
                 const std::shared_ptr<SLAMSystem>& SLAM,
                 const std::shared_ptr<inference_engine>& segmentation_engine,
                 const std::string& config_file_path) {
  // initialize TSDF
  auto TSDF = std::make_shared<TSDFSystem>(0.01, 0.06, 4, GetIntrinsicsFromFile(config_file_path),
                                           GetExtrinsicsFromFile(config_file_path));
  SLAM->startup();

  auto POSE_MANAGER = std::make_shared<pose_manager>();

  ImageRenderer renderer("tsdf", std::bind(&pose_manager::get_latest_pose, POSE_MANAGER), TSDF,
                         GetIntrinsicsFromFile(config_file_path));

  std::thread t_slam([&]() {
    while (true) {
      cv::Mat img_left, img_right;
      if (SLAM->terminate_is_requested()) break;
      // get sensor readings
      const int64_t timestamp = zed_native.GetStereoFrame(&img_left, &img_right);
      // visual slam
      const pose_valid_tuple m =
          SLAM->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
      const SE3<float> posecam_P_world(m.first.cast<float>().eval());
      if (m.second) POSE_MANAGER->register_valid_pose(timestamp, posecam_P_world);
    }
  });

  std::thread t_tsdf([&]() {
    while (true) {
      cv::Mat img_rgb, img_depth;
      if (SLAM->terminate_is_requested()) break;
      const int64_t timestamp = l515.GetRGBDFrame(&img_rgb, &img_depth);
      const SE3<float> posecam_P_world = POSE_MANAGER->query_pose(timestamp);
      cv::resize(img_rgb, img_rgb, cv::Size(), .5, .5);
      cv::resize(img_depth, img_depth, cv::Size(), .5, .5);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / l515.DepthScale());
      std::vector<cv::Mat> prob_map = segmentation_engine->infer_one(img_rgb, false);
      TSDF->Integrate(posecam_P_world, img_rgb, img_depth, prob_map[0], prob_map[1]);
      spdlog::debug("[Main] Frame integration takes {} ms",
                    GetTimestamp<std::chrono::milliseconds>() - timestamp);
    }
  });

  renderer.Run();
  t_slam.join();
  t_tsdf.join();
  SLAM->shutdown();
}

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  auto seg_model_path =
      op.add<popl::Value<std::string>>("m", "model", "PyTorch JIT traced model path");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto device_id = op.add<popl::Value<int>>("", "devid", "camera device id", 0);

  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (!vocab_file_path->is_set() || !config_file_path->is_set() || !seg_model_path->is_set()) {
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
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  // initialize cameras
  ZEDNative zed_native(*cfg, device_id->value());
  L515 l515;
  // initialize slam
  YAML::Node yaml_node = YAML::LoadFile(config_file_path->value());
  int tsdf_width = yaml_node["tsdf.width"].as<int>();
  int tsdf_height = yaml_node["tsdf.height"].as<int>();
  auto SLAM = std::make_shared<SLAMSystem>(cfg, vocab_file_path->value());
  auto my_engine =
      std::make_shared<inference_engine>(seg_model_path->value(), tsdf_width, tsdf_height);
  reconstruct(zed_native, l515, SLAM, my_engine, config_file_path->value());

  return EXIT_SUCCESS;
}
