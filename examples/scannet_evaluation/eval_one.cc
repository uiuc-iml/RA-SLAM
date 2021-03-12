#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <string>
#include <vector>

#include "utils/cuda/errors.cuh"
#include "utils/cuda/vector.cuh"
#include "utils/gl/image.h"
#include "utils/time.hpp"
#include "utils/tsdf/voxel_tsdf.cuh"

struct LogEntry {
  int id;
  SE3<float> cam_T_world;
};

CameraIntrinsics<float> get_intrinsics(const YAML::Node& config) {
  return CameraIntrinsics<float>(config["Camera.fx"].as<float>(), config["Camera.fy"].as<float>(),
                                 config["Camera.cx"].as<float>(), config["Camera.cy"].as<float>());
}

SE3<float> get_extrinsics(const YAML::Node& config) {
  const auto extrinsics = config["Extrinsics"].as<std::vector<float>>(std::vector<float>());
  if (extrinsics.empty()) {
    return SE3<float>::Identity();
  }
  const Eigen::Matrix4f tmp =
      Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(extrinsics.data());
  return SE3<float>(tmp);
}

const std::vector<LogEntry> parse_log_entries(const std::string& logdir, const YAML::Node& config) {
  const std::string trajectory_path = logdir + "/trajectory.txt";
  const SE3<float> extrinsics = get_extrinsics(config);

  int id;
  float buff[12];

  std::vector<LogEntry> log_entries;
  std::ifstream fin(trajectory_path);
  while (fin >> id >> buff[0] >> buff[1] >> buff[2] >> buff[3] >> buff[4] >> buff[5] >> buff[6] >>
         buff[7] >> buff[8] >> buff[9] >> buff[10] >> buff[11]) {
    const Eigen::Matrix<float, 3, 4> tmp =
        Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(buff);
    log_entries.push_back({id, extrinsics * SE3<float>(tmp)});
  }

  return log_entries;
}

void get_images_by_id(int id, float depth_scale, cv::Mat* img_rgb, cv::Mat* img_depth,
                      cv::Mat* img_ht, cv::Mat* img_lt, const std::string& logdir) {
  const std::string rgb_path = logdir + "/" + std::to_string(id) + "_rgb.png";
  const std::string depth_path = logdir + "/" + std::to_string(id) + "_depth.png";
  const std::string ht_path = logdir + "/" + std::to_string(id) + "_ht.png";
  const std::string lt_path = logdir + "/" + std::to_string(id) + "_no_ht.png";

  *img_rgb = cv::imread(rgb_path);
  const cv::Mat img_depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  const cv::Mat img_ht_raw = cv::imread(ht_path, cv::IMREAD_UNCHANGED);
  const cv::Mat img_lt_raw = cv::imread(lt_path, cv::IMREAD_UNCHANGED);
  img_depth_raw.convertTo(*img_depth, CV_32FC1, 1. / depth_scale);
  if (!img_ht_raw.empty()) {
    img_ht_raw.convertTo(*img_ht, CV_32FC1, 1. / 65535);
    img_lt_raw.convertTo(*img_lt, CV_32FC1, 1. / 65535);
  } else {
    *img_ht = cv::Mat::zeros(img_depth->rows, img_depth->cols, img_depth->type());
    *img_lt = cv::Mat::ones(img_depth->rows, img_depth->cols, img_depth->type());
  }
}

class DatasetEvaluator {
 public:
  DatasetEvaluator(const std::string& logdir, const YAML::Node& config)
      :
        logdir_(logdir),
        tsdf_(0.01, 0.06),
        intrinsics_(get_intrinsics(config)),
        log_entries_(parse_log_entries(logdir, config)),
        depth_scale_(config["depthmap_factor"].as<float>()) {
    spdlog::debug("[RGBD Intrinsics] fx: {} fy: {} cx: {} cy: {}", intrinsics_.fx, intrinsics_.fy,
                  intrinsics_.cx, intrinsics_.cy);
  }

  void run_all() {
    spdlog::info("Starting evaluation! Total frame count: {}", log_entries_.size());
    while (cnt_ < log_entries_.size()) {
      const LogEntry& log_entry = log_entries_[(cnt_++) % log_entries_.size()];
      const auto st_img = GetTimestamp<std::chrono::milliseconds>();
      get_images_by_id(log_entry.id, depth_scale_, &img_rgb_, &img_depth_, &img_ht_, &img_lt_,
                        logdir_);
      const auto end_img = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("Image IO takes {} ms", end_img - st_img);
      cv::cvtColor(img_rgb_, img_rgb_, cv::COLOR_BGR2RGB);
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.Integrate(img_rgb_, img_depth_, img_ht_, img_lt_, 4, intrinsics_,
                      log_entry.cam_T_world);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("Integration takes {} ms", end - st);
    }
    // all images visited, terminate
    spdlog::info("Evaluation completed! Extracting TSDF now...");
    const auto voxel_pos_prob = tsdf_.GatherValidSemantic();
    spdlog::info("Visible TSDF blocks count: {}", voxel_pos_prob.size());
    std::ofstream fout("/tmp/data.bin", std::ios::out | std::ios::binary);
    fout.write((char*)voxel_pos_prob.data(), voxel_pos_prob.size() * sizeof(VoxelSpatialTSDFSEGM));
    fout.close();
  }

 private:
  int cnt_ = 0;
  TSDFGrid tsdf_;
  cv::Mat img_rgb_, img_depth_, img_ht_, img_lt_;
  const std::string logdir_;
  const CameraIntrinsics<float> intrinsics_;
  const std::vector<LogEntry> log_entries_;
  const float depth_scale_;
};

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto logdir = op.add<popl::Value<std::string>>("", "logdir", "directory to the log files");
  auto config = op.add<popl::Value<std::string>>("c", "config", "path to the config file");

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
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

  if (!logdir->is_set() || !config->is_set()) {
    spdlog::error("Invalid arguments");
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  const auto yaml_node = YAML::LoadFile(config->value());
  DatasetEvaluator evaluator(logdir->value(), yaml_node);
  evaluator.run_all();

  return EXIT_SUCCESS;
}
