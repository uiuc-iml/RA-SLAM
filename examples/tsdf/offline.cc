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
#include "utils/gl/renderer_base.h"
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

class ImageRenderer : public RendererBase {
 public:
  ImageRenderer(const std::string& name, const std::string& logdir, const YAML::Node& config)
      : RendererBase(name),
        logdir_(logdir),
        tsdf_(0.02, 0.12),
        intrinsics_(get_intrinsics(config)),
        log_entries_(parse_log_entries(logdir, config)),
        depth_scale_(config["depthmap_factor"].as<float>()) {
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 2;
    spdlog::debug("[RGBD Intrinsics] fx: {} fy: {} cx: {} cy: {}", intrinsics_.fx, intrinsics_.fy,
                  intrinsics_.cx, intrinsics_.cy);
  }

 protected:
  void DispatchInput() override {
    ImGuiIO& io = ImGui::GetIO();
    if (io.MouseWheel != 0) {
      follow_cam_ = false;
      const Eigen::Vector3f move_cam(0, 0, io.MouseWheel * .1);
      const Eigen::Quaternionf virtual_cam_R_world = virtual_cam_T_world_.GetR();
      const Eigen::Vector3f virtual_cam_t_world = virtual_cam_T_world_.GetT();
      virtual_cam_T_world_ = SE3<float>(virtual_cam_R_world, virtual_cam_t_world - move_cam);
    }
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(0) && tsdf_rgba_.GetWidth()) {
      follow_cam_ = false;
      const ImVec2 delta = ImGui::GetMouseDragDelta(0);
      const Eigen::Vector2f delta_img(delta.x / io.DisplaySize.x * tsdf_rgba_.GetWidth(),
                                      delta.y / io.DisplaySize.y * tsdf_rgba_.GetHeight());
      const Eigen::Vector2f pos_new_img(io.MousePos.x / io.DisplaySize.x * tsdf_rgba_.GetWidth(),
                                        io.MousePos.y / io.DisplaySize.y * tsdf_rgba_.GetHeight());
      const Eigen::Vector2f pos_old_img = pos_new_img - delta_img;
      const Eigen::Vector3f pos_new_cam = intrinsics_.Inverse() * pos_new_img.homogeneous();
      const Eigen::Vector3f pos_old_cam = intrinsics_.Inverse() * pos_old_img.homogeneous();
      const Eigen::Vector3f pos_new_norm_cam = pos_new_cam.normalized();
      const Eigen::Vector3f pos_old_norm_cam = pos_old_cam.normalized();
      const Eigen::Vector3f rot_axis_cross_cam = pos_new_norm_cam.cross(pos_old_norm_cam);
      const float theta = acos(pos_new_norm_cam.dot(pos_old_norm_cam));
      const Eigen::Quaternionf R(Eigen::AngleAxisf(theta, rot_axis_cross_cam.normalized()));
      const SE3<float> pose_cam1_T_cam2(R, Eigen::Vector3f::Zero());
      virtual_cam_T_world_ = pose_cam1_T_cam2.Inverse() * virtual_cam_T_world_old_;
    } else if (!io.WantCaptureMouse && ImGui::IsMouseDragging(2)) {
      follow_cam_ = false;
      const ImVec2 delta = ImGui::GetMouseDragDelta(2);
      const Eigen::Vector3f translation(delta.x, delta.y, 0);
      const Eigen::Vector3f t = virtual_cam_T_world_old_.GetT();
      const Eigen::Quaternionf R = virtual_cam_T_world_old_.GetR();
      virtual_cam_T_world_ = SE3<float>(R, t + translation * .01);
    } else {
      virtual_cam_T_world_old_ = virtual_cam_T_world_;
    }
  }

  void Render() override {
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    // GUI
    ImGui::Begin("Menu");
    if (!running_ && ImGui::Button("Start")) {
      running_ = true;
    } else if (running_ && ImGui::Button("Pause")) {
      running_ = false;
    }
    if (ImGui::Button("Follow Camera")) {
      follow_cam_ = true;
    }
    // compute
    if (running_) {
      const LogEntry& log_entry = log_entries_[(cnt_++) % log_entries_.size()];
      cam_T_world_ = log_entry.cam_T_world;
      get_images_by_id(log_entry.id, depth_scale_, &img_rgb_, &img_depth_, &img_ht_, &img_lt_,
                       logdir_);
      cv::imshow("rgb", img_rgb_);
      if (!tsdf_rgba_.GetHeight() || !tsdf_rgba_.GetWidth() || !tsdf_normal_.GetHeight() ||
          !tsdf_normal_.GetWidth()) {
        tsdf_rgba_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
        tsdf_normal_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
      }
      cv::cvtColor(img_rgb_, img_rgb_, cv::COLOR_BGR2RGB);
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.Integrate(img_rgb_, img_depth_, img_ht_, img_lt_, 4, intrinsics_,
                      log_entry.cam_T_world);
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      ImGui::Text("Integration takes %lu ms", end - st);
      img_depth_.convertTo(img_depth_, CV_32FC1, 1. / 4);
      cv::imshow("depth", img_depth_);
      cv::waitKey(1);
    }
    if (follow_cam_) {
      static float step = 0;
      ImGui::SliderFloat("behind actual camera", &step, 0.0f, 3.0f);
      virtual_cam_T_world_ =
          SE3<float>(cam_T_world_.GetR(), cam_T_world_.GetT() + Eigen::Vector3f(0, 0, step));
    }
    if (ImGui::Button("Save TSDF")) {
      const auto voxel_pos_tsdf = tsdf_.GatherValid();
      spdlog::debug("{}", voxel_pos_tsdf.size());
      std::ofstream fout("/tmp/data.bin", std::ios::out | std::ios::binary);
      fout.write((char*)voxel_pos_tsdf.data(), voxel_pos_tsdf.size() * sizeof(VoxelSpatialTSDF));
      fout.close();
    }
    if (ImGui::Button("Save Mesh")) {
      std::vector<Eigen::Vector3f> vertex_buffer;
      std::vector<Eigen::Vector3i> index_buffer;
      tsdf_.GatherValidMesh(&vertex_buffer, &index_buffer);
      std::ofstream vout("/tmp/vertices.bin", std::ios::out | std::ios::binary);
      std::ofstream iout("/tmp/indices.bin", std::ios::out | std::ios::binary);

      vout.write((char*)vertex_buffer.data(), vertex_buffer.size() * sizeof(Eigen::Vector3f));
      iout.write((char*)index_buffer.data(), index_buffer.size() * sizeof(Eigen::Vector3i));

      vout.close();
      iout.close();
    }
    // render
    if (!img_depth_.empty() && !img_rgb_.empty()) {
      const CameraParams virtual_cam(intrinsics_, img_depth_.rows, img_depth_.cols);
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.RayCast(10, virtual_cam, virtual_cam_T_world_, &tsdf_rgba_, &tsdf_normal_);
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      ImGui::Text("Rendering takes %lu ms", end - st);
      static int render_mode = 1;
      ImGui::RadioButton("rgb", &render_mode, 0);
      ImGui::SameLine();
      ImGui::RadioButton("normal", &render_mode, 1);
      if (render_mode == 0) {
        tsdf_rgba_.Draw();
      } else if (render_mode == 1) {
        tsdf_normal_.Draw();
      }
    }
    ImGui::End();
  }

 private:
  int cnt_ = 0;
  bool running_ = false;
  bool follow_cam_ = true;
  GLImage8UC4 tsdf_rgba_;
  GLImage8UC4 tsdf_normal_;
  TSDFGrid tsdf_;
  cv::Mat img_rgb_, img_depth_, img_ht_, img_lt_;
  SE3<float> cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_old_ = SE3<float>::Identity();
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
  ImageRenderer renderer("tsdf", logdir->value(), yaml_node);
  renderer.Run();

  return EXIT_SUCCESS;
}
