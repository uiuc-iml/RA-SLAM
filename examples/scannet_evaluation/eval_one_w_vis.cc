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

#include "segmentation/inference.h"
#include "utils/cuda/errors.cuh"
#include "utils/cuda/vector.cuh"
#include "utils/gl/image.h"
#include "utils/gl/renderer_base.h"
#include "utils/offline_data_provider/scannet_sens_reader.h"
#include "utils/time.hpp"
#include "utils/tsdf/voxel_tsdf.cuh"

class ImageRenderer : public RendererBase {
 public:
  ImageRenderer(const std::string& name, const std::string& segm_model_path,
                const std::string& sens_path)
      : RendererBase(name),
        sens_reader_(sens_path),
        tsdf_(0.01, 0.06),
        intrinsics_(sens_reader_.get_camera_intrinsics()),
        depth_scale_(sens_reader_.get_depth_map_factor()),
        segm_(segm_model_path, sens_reader_.get_width(), sens_reader_.get_height()) {
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
      if (cnt_ == sens_reader_.get_size()) {
        // all images visited, terminate
        spdlog::info("Evaluation completed! Extracting TSDF now...");
        const auto voxel_pos_prob = tsdf_.GatherValidSemantic();
        spdlog::info("Visible TSDF blocks count: {}", voxel_pos_prob.size());
        std::ofstream fout("/tmp/data.bin", std::ios::out | std::ios::binary);
        fout.write((char*)voxel_pos_prob.data(),
                   voxel_pos_prob.size() * sizeof(VoxelSpatialTSDFSEGM));
        fout.close();
        exit(0);
      }
      cam_T_world_ = sens_reader_.get_camera_pose_by_id(cnt_);
      sens_reader_.get_color_frame_by_id(&img_rgb_, cnt_);
      sens_reader_.get_depth_frame_by_id(&img_depth_, cnt_);
      img_depth_.convertTo(img_depth_, CV_32FC1, 1. / depth_scale_);
      // use seg engine to get ht/lt img
      cv::imshow("bgr", img_rgb_);
      if (!tsdf_rgba_.GetHeight() || !tsdf_rgba_.GetWidth() || !tsdf_normal_.GetHeight() ||
          !tsdf_normal_.GetWidth()) {
        tsdf_rgba_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
        tsdf_normal_.BindImage(img_depth_.rows, img_depth_.cols, nullptr);
      }
      std::vector<cv::Mat> prob_map = segm_.infer_one(img_rgb_, false);
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.Integrate(img_rgb_, img_depth_, prob_map[0], prob_map[1], 5, intrinsics_, cam_T_world_);
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      ImGui::Text("Integration takes %lu ms", end - st);
      img_depth_.convertTo(img_depth_, CV_32FC1, 1. / 4);
      cv::imshow("depth", img_depth_);
      cv::waitKey(1);
      cnt_++;
    }
    if (follow_cam_) {
      static float step = 0;
      ImGui::SliderFloat("behind actual camera", &step, 0.0f, 3.0f);
      virtual_cam_T_world_ =
          SE3<float>(cam_T_world_.GetR(), cam_T_world_.GetT() + Eigen::Vector3f(0, 0, step));
    }
    // render
    if (!img_depth_.empty() && !img_rgb_.empty()) {
      const CameraParams virtual_cam(intrinsics_, img_depth_.rows, img_depth_.cols);
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      tsdf_.RayCast(10, virtual_cam, virtual_cam_T_world_, &tsdf_rgba_, &tsdf_normal_);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
  scannet_sens_reader sens_reader_;
  TSDFGrid tsdf_;
  cv::Mat img_rgb_, img_depth_, img_ht_, img_lt_;
  SE3<float> cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_old_ = SE3<float>::Identity();
  const CameraIntrinsics<float> intrinsics_;
  const float depth_scale_;
  inference_engine segm_;
};

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto model = op.add<popl::Value<std::string>>("", "model", "path PyTorch JIT traced model");
  auto sens =
      op.add<popl::Value<std::string>>("", "sens", "path to scannet sensor stream .sens file");

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

  if (!model->is_set() || !sens->is_set()) {
    spdlog::error("Invalid arguments");
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  ImageRenderer renderer("tsdf", model->value(), sens->value());
  renderer.Run();

  return EXIT_SUCCESS;
}