#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

#include <pangolin_viewer/viewer.h>

#include <openvslam/config.h>

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "cameras/zed.h"
#include "modules/slam_module.h"
#include "utils/data_logger.hpp"
#include "utils/time.hpp"

class DepthData {
 public:
  DepthData() = default;

  DepthData(const DepthData &others)
    : img_rgb(others.img_rgb.clone()),
      img_depth(others.img_depth.clone()),
      id(others.id) {}

  cv::Mat img_rgb;
  cv::Mat img_depth;
  unsigned int id;
};

class DepthLogger : public DataLogger<DepthData> {
 public:
  DepthLogger(const std::string &logdir)
      : logdir_(logdir),
        DataLogger<DepthData>() {}

  std::vector<unsigned int> logged_ids;

 protected:
  void save_data(const DepthData &data) override {
    const std::string rgb_path = logdir_ + "/" + std::to_string(data.id) + "_rgb.png";
    const std::string depth_path = logdir_ + "/" + std::to_string(data.id) + "_depth.png";
    cv::Mat img_depth_uint16;
    data.img_depth.convertTo(img_depth_uint16, CV_16UC1);
    cv::imwrite(rgb_path, data.img_rgb);
    cv::imwrite(depth_path, img_depth_uint16);
    logged_ids.push_back(data.id);
  }

 private:
  const std::string logdir_;
};

void tracking(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path,
              const std::string &map_db_path,
              const std::string &logdir,
              ZED *camera) {
  SLAMSystem SLAM(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  DepthLogger logger(logdir);

  DepthData data;
  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    cv::Mat img_left, img_right;
    while (true) {
      if (SLAM.terminate_is_requested())
        break;

      camera->get_stereo_img(&img_left, &img_right, &data.img_rgb, &data.img_depth);
      const auto timestamp = get_timestamp<std::chrono::microseconds>();

      data.id = SLAM.feed_stereo_images(img_left, img_right, timestamp / 1e6);

      logger.log_data(data);
    }

    while (SLAM.loop_BA_is_running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
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

std::shared_ptr<openvslam::config> get_config(const std::string &config_file_path,
                                              const ZED &camera) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  // modify configuration based on realsense camera data
  // pre-defined stream profile
  auto cam_config = camera.get_camera_config();
  yaml_node["Camera.fps"] = cam_config.fps;
  yaml_node["Camera.cols"] = cam_config.resolution.width;
  yaml_node["Camera.rows"] = cam_config.resolution.height;
  yaml_node["Camera.color_order"] = "Gray";
  // camera intrinsics
  yaml_node["Camera.fx"] = cam_config.calibration_parameters.left_cam.fx;
  yaml_node["Camera.fy"] = cam_config.calibration_parameters.left_cam.fy;
  yaml_node["Camera.cx"] = cam_config.calibration_parameters.left_cam.cx;
  yaml_node["Camera.cy"] = cam_config.calibration_parameters.left_cam.cy;
  yaml_node["Camera.focal_x_baseline"] =
    cam_config.calibration_parameters.stereo_transform.getTranslation().x *
    cam_config.calibration_parameters.left_cam.fx / 1e3; // unit [mm] to [m]
  // zero camera distortion
  yaml_node["Camera.k1"] = 0;
  yaml_node["Camera.k2"] = 0;
  yaml_node["Camera.p1"] = 0;
  yaml_node["Camera.p2"] = 0;
  yaml_node["Camera.k3"] = 0;

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
  auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db",
                            "path to store the map database", "");
  auto log_dir = op.add<popl::Value<std::string>>("", "logdir",
                            "directory to store logged data", "./log");
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

  ZED camera;

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = get_config(config_file_path->value(), camera);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  tracking(cfg,
      vocab_file_path->value(), map_db_path->value(), log_dir->value(), &camera);

  return EXIT_SUCCESS;
}
