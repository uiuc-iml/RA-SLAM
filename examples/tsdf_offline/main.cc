#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "cameras/zed.h"
#include "third_party/popl/include/popl.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/time.hpp"
#include "utils/tsdf/voxel_tsdf.cuh"

CameraIntrinsics<float> get_zed_intrinsics() {
  ZED camera;
  const auto cam_config = camera.get_camera_config();
  const CameraIntrinsics<float> intrinsics(cam_config.calibration_parameters.left_cam.fx,
                                           cam_config.calibration_parameters.left_cam.fy,
                                           cam_config.calibration_parameters.left_cam.cx,
                                           cam_config.calibration_parameters.left_cam.cy);
  return intrinsics;
}

struct LogEntry {
  int id;
  SE3<float> cam_P_world;
};

const std::vector<LogEntry> parse_log_entries(const std::string &logdir) {
  const std::string trajectory_path = logdir + "/trajectory.txt";
  int id;
  float m00, m01, m02, m03;
  float m10, m11, m12, m13;
  float m20, m21, m22, m23;

  std::vector<LogEntry> log_entries;
  std::ifstream fin(trajectory_path);
  while (fin >> id >> m00 >> m01 >> m02 >> m03
                   >> m10 >> m11 >> m12 >> m13
                   >> m20 >> m21 >> m22 >> m23) {
    log_entries.push_back({id, SE3<float>(m00, m01, m02, m03, 
                                          m10, m11, m12, m13,
                                          m20, m21, m22, m23,
                                          0, 0, 0, 1)});
  }

  return log_entries;
}

void get_images_by_id(int id, cv::Mat *img_rgb, cv::Mat *img_depth, 
                      const std::string &logdir) {
  const std::string rgb_path = logdir + "/" + std::to_string(id) + "_rgb.png";
  const std::string depth_path = logdir + "/" + std::to_string(id) + "_depth.png";

  *img_rgb = cv::imread(rgb_path);
  const cv::Mat img_depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  img_depth_raw.convertTo(*img_depth, CV_32FC1, 1./1000);
}

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto logdir = op.add<popl::Value<std::string>>("", "logdir", "directory to the log files");

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  try {
    op.parse(argc, argv);
  } catch (const std::exception &e) {
    spdlog::error(e.what());
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (debug_mode->is_set())
    spdlog::set_level(spdlog::level::debug);
  else
    spdlog::set_level(spdlog::level::info);

  if (!logdir->is_set()) {
    spdlog::error("Invalid arguments");
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  TSDFGrid tsdf(0.01, 0.08, 5);

  const auto intrinsics = get_zed_intrinsics();
  const auto log_entries = parse_log_entries(logdir->value());
  cv::Mat img_rgb, img_depth, img_tsdf;
  for (int i = 0; i < log_entries.size(); ++i) {
    const auto &log_entry = log_entries[i];
    get_images_by_id(log_entry.id, &img_rgb, &img_depth, logdir->value());
    cv::imshow("rgb", img_rgb);
    cv::imshow("depth", img_depth);
    tsdf.Integrate(img_rgb, img_depth, intrinsics, log_entry.cam_P_world);
    if (img_tsdf.empty()) {
      img_depth.copyTo(img_tsdf);
    }
    tsdf.RayCast(&img_tsdf, intrinsics, log_entry.cam_P_world);
    cv::imshow("tsdf", img_tsdf);
    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}
