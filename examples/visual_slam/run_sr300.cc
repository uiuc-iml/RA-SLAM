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

#include "cameras/sr300.h"
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
              const SR300 &camera,
              const std::string &map_db_path,
              const std::string &logdir,
              bool use_depth = true) {
  SLAMSystem SLAM(cfg, vocab_file_path);
  SLAM.startup();

  pangolin_viewer::viewer viewer(
      cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());

  DepthLogger logger(logdir);

  std::thread t([&]() {
    const auto start = std::chrono::steady_clock::now();
    DepthData data;
    while (true) {
      if (SLAM.terminate_is_requested())
        break;

      camera.get_rgbd_frame(&data.img_rgb, &data.img_depth);
      const auto timestamp = get_timestamp<std::chrono::microseconds>();

      if (use_depth)
        data.id = SLAM.feed_rgbd_images(data.img_rgb, data.img_depth, timestamp / 1e6);
      else
        SLAM.feed_monocular_frame(data.img_rgb, timestamp / 1e6);

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

std::shared_ptr<openvslam::config> get_and_set_config(const std::string &config_file_path,
                                                      SR300 *camera) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  // modify configuration based on realsense camera data
  // pre-defined stream profile
  yaml_node["Camera.fps"] = SR300::FPS;
  yaml_node["Camera.cols"] = SR300::WIDTH;
  yaml_node["Camera.rows"] = SR300::HEIGHT;
  yaml_node["Camera.color_order"] = "RGB";
  // camera intrinsics
  rs2_intrinsics i = camera->get_camera_intrinsics();
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
  yaml_node["depthmap_factor"] = camera->get_depth_scale();
  // realsense depth camera options
  camera->set_depth_sensor_option(RS2_OPTION_FILTER_OPTION,
                                  yaml_node["RS.filter_option"].as<float>());
  camera->set_depth_sensor_option(RS2_OPTION_MOTION_RANGE,
                                  yaml_node["RS.motion_range"].as<float>());

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
  auto depth = op.add<popl::Switch>("", "depth", "use depth information");
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

  SR300 camera;

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = get_and_set_config(config_file_path->value(), &camera);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  tracking(cfg,
      vocab_file_path->value(), camera, map_db_path->value(), log_dir->value(), depth->is_set());

  return EXIT_SUCCESS;
}
