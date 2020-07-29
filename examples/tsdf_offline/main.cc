#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "cameras/zed.h"
#include "imgui.h"
#include "utils/gl/image.h"
#include "utils/gl/renderer_base.h"
#include "third_party/popl/include/popl.hpp"
#include "utils/cuda/vector.cuh"
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

class ImageRenderer : public RendererBase {
 public:
  ImageRenderer(const std::string &name, const std::string &logdir, int img_h, int img_w)
     : RendererBase(name), logdir_(logdir),
       tsdf_(0.01, 0.06),
       intrinsics_(get_zed_intrinsics()),
       log_entries_(parse_log_entries(logdir)) {
    ImGuiIO &io = ImGui::GetIO();
    io.FontGlobalScale = 2;
  }

 protected:
  void DispatchInput() override {
    ImGuiIO &io = ImGui::GetIO();
    if (io.MouseWheel != 0) {
      follow_cam_ = false;
      const Vector3<float> move_cam(0, 0, io.MouseWheel * .1);
      const SO3<float> virtual_cam_R_world = virtual_cam_P_world_.GetR(); 
      const Vector3<float> virtual_cam_T_world = virtual_cam_P_world_.GetT();
      virtual_cam_P_world_ = SE3<float>(virtual_cam_R_world, virtual_cam_T_world - move_cam);
    }
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(0)) {
      follow_cam_ = false;
      const ImVec2 delta = ImGui::GetMouseDragDelta(0);
      const Vector2<float> delta_img(delta.x, delta.y);
      const Vector2<float> pos_new_img(io.MousePos.x, io.MousePos.y);
      const Vector2<float> pos_old_img = pos_new_img - delta_img;
      const Vector3<float> pos_new_cam = intrinsics_.Inverse() * Vector3<float>(pos_new_img);
      const Vector3<float> pos_old_cam = intrinsics_.Inverse() * Vector3<float>(pos_old_img);
      const Vector3<float> pos_new_norm_cam = pos_new_cam / sqrt(pos_new_cam.dot(pos_new_cam));
      const Vector3<float> pos_old_norm_cam = pos_old_cam / sqrt(pos_old_cam.dot(pos_old_cam));
      const Vector3<float> rot_axis_cross_cam = pos_new_norm_cam.cross(pos_old_norm_cam);
      const float theta = acos(pos_new_norm_cam.dot(pos_old_norm_cam));
      const Vector3<float> w = rot_axis_cross_cam / sin(theta) * theta;
      const Matrix3<float> w_x(0, -w.z, w.y, w.z, 0, -w.x, -w.y, w.x, 0);
      const Matrix3<float> R = Matrix3<float>::Identity() + 
                               sin(theta) / theta * w_x + 
                               (1 - cos(theta)) / (theta * theta) * w_x * w_x;
      const SE3<float> pose_cam1_P_cam2(R, Vector3<float>(0));
      virtual_cam_P_world_ = pose_cam1_P_cam2.Inverse() * virtual_cam_P_world_old_;
    }
    else if (!io.WantCaptureMouse && ImGui::IsMouseDragging(2)) {
      follow_cam_ = false;
      const ImVec2 delta = ImGui::GetMouseDragDelta(2);
      const Vector3<float> translation(delta.x, delta.y, 0);
      const Vector3<float> T = virtual_cam_P_world_old_.GetT();
      const Matrix3<float> R = virtual_cam_P_world_old_.GetR();
      virtual_cam_P_world_ = SE3<float>(R, T + translation * .01);
    }
    else {
      virtual_cam_P_world_old_ = virtual_cam_P_world_;
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
    if (!running_ && ImGui::Button("Start")) { running_ = true; }
    else if (running_ && ImGui::Button("Pause")) { running_ = false; }
    if (ImGui::Button("Follow Camera")) { follow_cam_ = true; }
    // compute
    if (running_) {
      const LogEntry &log_entry = log_entries_[(cnt_++) % log_entries_.size()];
      cam_P_world_ = log_entry.cam_P_world;
      get_images_by_id(log_entry.id, &img_rgb_, &img_depth_, logdir_);
      cv::imshow("rgb", img_rgb_);
      cv::imshow("depth", img_depth_);
      if (!tsdf_rgba_.height || !tsdf_rgba_.width || !tsdf_normal_.height || !tsdf_normal_.width) {
        tsdf_rgba_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
        tsdf_normal_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
      }
      cv::waitKey(1);
      cv::cvtColor(img_rgb_, img_rgb_, cv::COLOR_BGR2RGB);
      tsdf_.Integrate(img_rgb_, img_depth_, 3, intrinsics_, log_entry.cam_P_world);
    }
    if (follow_cam_) { virtual_cam_P_world_ = cam_P_world_; }
    // render
    if (!img_depth_.empty() && !img_rgb_.empty()) {
      const CameraParams virtual_cam(intrinsics_, img_depth_.rows, img_depth_.cols);
      tsdf_.RayCast(10, virtual_cam, virtual_cam_P_world_, &tsdf_rgba_, &tsdf_normal_);
      static int render_mode = 0;
      ImGui::RadioButton("rgb", &render_mode, 0); ImGui::SameLine();
      ImGui::RadioButton("normal", &render_mode, 1);
      if (render_mode == 0) { tsdf_rgba_.Draw(); }
      else if (render_mode == 1) { tsdf_normal_.Draw(); }
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
  cv::Mat img_rgb_, img_depth_;
  SE3<float> cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_old_ = SE3<float>::Identity();
  const std::string logdir_;
  const CameraIntrinsics<float> intrinsics_;
  const std::vector<LogEntry> log_entries_;
};

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

  ImageRenderer renderer("tsdf", logdir->value(), 100, 100);
  renderer.Run();

  return EXIT_SUCCESS;
}
