#include "utils/offline_data_provider/folder_reader.h"

#include <assert.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>

folder_reader::folder_reader(const string& folder_path) {
  logdir_ = folder_path;

  /* Get config */
  camera_config_ = YAML::LoadFile(folder_path + "/camera_config.yaml");

  /* Parse number of entries */
  log_entries_ = parse_log_entries();
  size_ = (int)(log_entries_.size());

  /* Infer width/height and cache it for easier access */
  cv::Mat test_rgb, test_depth;
  get_depth_frame_by_id(&test_depth, 0);
  get_color_frame_by_id(&test_rgb, size_ - 1);
  assert(test_depth.cols == test_rgb.cols);
  assert(test_depth.rows == test_rgb.rows);
  width_ = test_depth.cols;
  height_ = test_depth.rows;
  depth_factor_ = camera_config_["depthmap_factor"].as<float>();
}

CameraIntrinsics<float> folder_reader::get_camera_intrinsics() {
  float fx = camera_config_["Camera.fx"].as<float>();
  float fy = camera_config_["Camera.fy"].as<float>();
  float cx = camera_config_["Camera.cx"].as<float>();
  float cy = camera_config_["Camera.cy"].as<float>();
  return CameraIntrinsics<float>(fx, fy, cx, cy);
}

SE3<float> folder_reader::get_camera_extrinsics() {
  const auto extrinsics = camera_config_["Extrinsics"].as<std::vector<float>>(std::vector<float>());

  /* If extrinsic was not provided, assume identity */
  if (extrinsics.empty()) {
    return SE3<float>::Identity();
  }

  /* use Eigen to format std vector to SE3 */
  const Eigen::Matrix4f tmp =
      Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(extrinsics.data());
  return SE3<float>(tmp);
}

float folder_reader::get_depth_map_factor() { return depth_factor_; }

void folder_reader::get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  const std::string depth_path = logdir_ + "/" + std::to_string(frame_idx) + "_depth.png";
  *depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
}

void folder_reader::get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  const std::string rgb_path = logdir_ + "/" + std::to_string(frame_idx) + "_rgb.png";
  *rgb_img = cv::imread(rgb_path);
  cv::cvtColor(*rgb_img, *rgb_img, cv::COLOR_BGR2RGB);
}

SE3<float> folder_reader::get_camera_pose_by_id(int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  return log_entries_[frame_idx].cam_T_world;
}

int folder_reader::get_size() { return size_; }

int folder_reader::get_width() { return width_; }

int folder_reader::get_height() { return height_; }

std::vector<LogEntry> folder_reader::parse_log_entries() {
  const std::string trajectory_path = logdir_ + "/trajectory.txt";
  const SE3<float> extrinsics = this->get_camera_extrinsics();

  int id;
  float buff[12];

  std::vector<LogEntry> log_entries;

  /* Shift read SE3 value. The last row (0 0 0 1) is not saved */
  std::ifstream fin(trajectory_path);
  while (fin >> id >> buff[0] >> buff[1] >> buff[2] >> buff[3] >> buff[4] >> buff[5] >> buff[6] >>
         buff[7] >> buff[8] >> buff[9] >> buff[10] >> buff[11]) {
    const Eigen::Matrix<float, 3, 4> tmp =
        Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(buff);
    log_entries.push_back({id, extrinsics * SE3<float>(tmp)});
  }

  return log_entries;
}
