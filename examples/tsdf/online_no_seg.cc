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

std::shared_ptr<openvslam::config> get_and_set_config(const std::string &config_file_path) {
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  const stereo_rectifier rectifier(yaml_node);
  const cv::Mat rectified_intrinsics = rectifier.get_rectified_intrinsics();
  yaml_node["Camera.fx"] = rectified_intrinsics.at<double>(0, 0);
  yaml_node["Camera.fy"] = rectified_intrinsics.at<double>(1, 1);
  yaml_node["Camera.cx"] = rectified_intrinsics.at<double>(0, 2);
  yaml_node["Camera.cy"] = rectified_intrinsics.at<double>(1, 2);
  yaml_node["Camera.focal_x_baseline"] = -rectified_intrinsics.at<double>(0, 3);
  return std::make_shared<openvslam::config>(yaml_node);
}

CameraIntrinsics<float> get_intrinsics(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  return CameraIntrinsics<float>(config["Camera.fx"].as<float>(),
                                 config["Camera.fy"].as<float>(),
                                 config["Camera.cx"].as<float>(),
                                 config["Camera.cy"].as<float>());
}

SE3<float> get_extrinsics(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  const auto m = config["Extrinsics"].as<std::vector<double>>();
  return SE3<float>(
    m[0], m[1], m[2], m[3],
    m[4], m[5], m[6], m[7],
    m[8], m[9], m[10], m[11],
    m[12], m[13], m[14], m[15]
  );
}

class ImageRenderer : public RendererBase {
 public:
  ImageRenderer(const std::string &name,
                const std::shared_ptr<SLAMSystem> &slam,
                const std::shared_ptr<TSDFSystem> &tsdf,
                const std::string &config_file_path)
     : RendererBase(name),
       slam_(slam),
       tsdf_(tsdf),
       map_publisher_(slam->get_map_publisher()),
       config_(YAML::LoadFile(config_file_path)),
       virtual_cam_(get_intrinsics(config_file_path), 360, 640) {
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
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(0) && tsdf_normal_.width) {
      follow_cam_ = false;
      const ImVec2 delta = ImGui::GetMouseDragDelta(0);
      const Vector2<float> delta_img(delta.x / io.DisplaySize.x * tsdf_normal_.width,
                                     delta.y / io.DisplaySize.y * tsdf_normal_.height);
      const Vector2<float> pos_new_img(io.MousePos.x / io.DisplaySize.x * tsdf_normal_.width,
                                       io.MousePos.y / io.DisplaySize.y * tsdf_normal_.height);
      const Vector2<float> pos_old_img = pos_new_img - delta_img;
      const Vector3<float> pos_new_cam = virtual_cam_.intrinsics_inv * Vector3<float>(pos_new_img);
      const Vector3<float> pos_old_cam = virtual_cam_.intrinsics_inv * Vector3<float>(pos_old_img);
      const Vector3<float> pos_new_norm_cam = pos_new_cam / sqrt(pos_new_cam.dot(pos_new_cam));
      const Vector3<float> pos_old_norm_cam = pos_old_cam / sqrt(pos_old_cam.dot(pos_old_cam));
      const Vector3<float> rot_axis_cross_cam = pos_new_norm_cam.cross(pos_old_norm_cam);
      const float theta = acos(pos_new_norm_cam.dot(pos_old_norm_cam));
      const Vector3<float> w = rot_axis_cross_cam / sin(theta) * theta;
      const Matrix3<float> w_x(0, -w.z, w.y, w.z, 0, -w.x, -w.y, w.x, 0);
      const Matrix3<float> R = Matrix3<float>::Identity() +
                               (float)sin(theta) / theta * w_x +
                               (float)(1 - cos(theta)) / (theta * theta) * w_x * w_x;
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
    if (ImGui::Button("Follow Camera")) { follow_cam_ = true; }
    // compute
    const auto m = map_publisher_->get_current_cam_pose();
    cam_P_world_ = SE3<float>(
      m(0, 0), m(0, 1), m(0, 2), m(0, 3),
      m(1, 0), m(1, 1), m(1, 2), m(1, 3),
      m(2, 0), m(2, 1), m(2, 2), m(2, 3),
      m(3, 0), m(3, 1), m(3, 2), m(3, 3)
    );
    if (!tsdf_normal_.height || !tsdf_normal_.width) {
      tsdf_normal_.BindImage(virtual_cam_.img_h, virtual_cam_.img_w, nullptr);
    }
    if (follow_cam_) {
      static float step = 0;
      ImGui::SliderFloat("behind actual camera", &step, 0.0f, 3.0f);
      virtual_cam_P_world_ = SE3<float>(cam_P_world_.GetR(),
                                        cam_P_world_.GetT() + Vector3<float>(0, 0, step));
    }
    // render
    const auto st = get_timestamp<std::chrono::milliseconds>();
    tsdf_->Render(virtual_cam_, virtual_cam_P_world_, &tsdf_normal_);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto end = get_timestamp<std::chrono::milliseconds>();
    ImGui::Text("Rendering takes %lu ms", end - st);
    tsdf_normal_.Draw();
    ImGui::End();
  }

  void RenderExit() override {
    slam_->request_terminate();
  }

 private:
  bool follow_cam_ = true;
  GLImage8UC4 tsdf_normal_;
  std::shared_ptr<SLAMSystem> slam_;
  std::shared_ptr<TSDFSystem> tsdf_;
  std::shared_ptr<openvslam::publish::map_publisher> map_publisher_;
  SE3<float> cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_old_ = SE3<float>::Identity();
  const YAML::Node config_;
  const CameraParams virtual_cam_;
};

void reconstruct(const ZEDNative &zed_native, const L515 &l515,
                 const std::shared_ptr<SLAMSystem> &SLAM,
                 const std::string &config_file_path) {
  // initialize TSDF
  auto TSDF = std::make_shared<TSDFSystem>(0.01, 0.06, 4,
      get_intrinsics(config_file_path), get_extrinsics(config_file_path));
  SLAM->startup();

  ImageRenderer renderer("tsdf", SLAM, TSDF, config_file_path);

  auto POSE_MANAGER = std::make_shared<pose_manager>();

  std::thread t_slam([&]() {
    cv::Mat img_left, img_right;
    while (true) {
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
