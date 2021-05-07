#include "modules/renderer_module.h"

#include "utils/config_reader.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/time.hpp"

ImageRenderer::ImageRenderer(const std::string& name,
                             std::function<SE3<float>()> get_latest_pose_func,
                             const std::shared_ptr<TSDFSystem>& tsdf,
                             const CameraIntrinsics<float> intrinsics)
    : RendererBase(name),
      tsdf_(tsdf),
      get_latest_pose_func_(get_latest_pose_func),
      virtual_cam_(intrinsics, 480, 640) {
  ImGuiIO& io = ImGui::GetIO();

  // Set font size
  io.FontGlobalScale = 2;

  // Initialize OpenGL drawer
  if (!tsdf_normal_.GetHeight() || !tsdf_normal_.GetWidth()) {
    tsdf_normal_.BindImage(virtual_cam_.img_h, virtual_cam_.img_w, nullptr);
  }
  if (!tsdf_rgba_.GetHeight() || !tsdf_rgba_.GetWidth()) {
    tsdf_rgba_.BindImage(virtual_cam_.img_h, virtual_cam_.img_w, nullptr);
  }
}

void ImageRenderer::DispatchInput() {
  ImGuiIO& io = ImGui::GetIO();
  if (io.MouseWheel != 0) {
    // Scroll wheel is scrolled
    HandleMouseScrollWheel(io);
  }
  if (!io.WantCaptureMouse && ImGui::IsMouseDragging(0) && tsdf_rgba_.GetWidth()) {
    // Left click is pressed and dragged
    HandleMouseLeftClick(io);
  } else if (!io.WantCaptureMouse && ImGui::IsMouseDragging(2)) {
    // Scroll wheel is clicked, pressed and dragged
    HandleMouseMiddleClick(io);
  } else {
    // No meaningful input from mouse is detected
    virtual_cam_T_world_old_ = virtual_cam_T_world_;
  }
}

void ImageRenderer::Render() {
  cam_T_world_ = get_latest_pose_func_();
  InitializeGUI();
  DefineGUIButton();
  DefineGUISlider();
  DefineGUITexture();

  // Actual rendering
  const auto st = GetTimestamp<std::chrono::milliseconds>();
  tsdf_->Render(virtual_cam_, virtual_cam_T_world_, &tsdf_rgba_, &tsdf_normal_, raycast_depth);
  const auto end = GetTimestamp<std::chrono::milliseconds>();
  ImGui::Text("Render lag: %lu ms", end - st);
  if (render_mode == 0) {
    tsdf_rgba_.Draw();
  } else if (render_mode == 1) {
    tsdf_normal_.Draw();
  }

  // End of rendering cycle cleanup
  ImGui::End();

  // In order to prevent ray casting to be run for excessive amount of time,
  // we place a manual delay in the renderer
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  if (tsdf_->is_terminated()) {
    RenderExit();
  }
}

void ImageRenderer::RenderExit() { running_ = false; }

void ImageRenderer::HandleMouseScrollWheel(const ImGuiIO& io) {
  follow_cam_ = false;

  // Move virtual camera forward/backward
  const Eigen::Vector3f move_cam(0, 0, io.MouseWheel * .1);
  const Eigen::Quaternionf virtual_cam_R_world = virtual_cam_T_world_.GetR();
  const Eigen::Vector3f virtual_cam_t_world = virtual_cam_T_world_.GetT();
  virtual_cam_T_world_ = SE3<float>(virtual_cam_R_world, virtual_cam_t_world - move_cam);
}

void ImageRenderer::HandleMouseLeftClick(const ImGuiIO& io) {
  follow_cam_ = false;

  // Rotate Virtual Camera
  const ImVec2 delta = ImGui::GetMouseDragDelta(0);
  const Eigen::Vector2f delta_img(delta.x / io.DisplaySize.x * tsdf_rgba_.GetWidth(),
                                  delta.y / io.DisplaySize.y * tsdf_rgba_.GetHeight());
  const Eigen::Vector2f pos_new_img(io.MousePos.x / io.DisplaySize.x * tsdf_rgba_.GetWidth(),
                                    io.MousePos.y / io.DisplaySize.y * tsdf_rgba_.GetHeight());
  const Eigen::Vector2f pos_old_img = pos_new_img - delta_img;
  const Eigen::Vector3f pos_new_cam = virtual_cam_.intrinsics_inv * pos_new_img.homogeneous();
  const Eigen::Vector3f pos_old_cam = virtual_cam_.intrinsics_inv * pos_old_img.homogeneous();
  const Eigen::Vector3f pos_new_norm_cam = pos_new_cam.normalized();
  const Eigen::Vector3f pos_old_norm_cam = pos_old_cam.normalized();
  const Eigen::Vector3f rot_axis_cross_cam = pos_new_norm_cam.cross(pos_old_norm_cam);
  const float theta = acos(pos_new_norm_cam.dot(pos_old_norm_cam));
  const Eigen::Quaternionf R(Eigen::AngleAxisf(theta, rot_axis_cross_cam.normalized()));
  const SE3<float> pose_cam1_T_cam2(R, Eigen::Vector3f::Zero());
  virtual_cam_T_world_ = pose_cam1_T_cam2.Inverse() * virtual_cam_T_world_old_;
}

void ImageRenderer::HandleMouseMiddleClick(const ImGuiIO& io) {
  // User wants to translate the viewplane, disable CamFollow
  follow_cam_ = false;

  // Translate mouse motion to viewplane change
  const ImVec2 delta = ImGui::GetMouseDragDelta(2);
  const Eigen::Vector3f translation(delta.x, delta.y, 0);
  const Eigen::Vector3f T = virtual_cam_T_world_old_.GetT();
  const Eigen::Quaternionf R = virtual_cam_T_world_old_.GetR();

  // Scale mouse delta by .01
  virtual_cam_T_world_ = SE3<float>(R, T + translation * .01);
}

void ImageRenderer::DefineGUIButton() {
  // Width of the button is set to fill the menu
  // When button height is set to 0, ImGUI will infer it.
  ImVec2 button_size(display_w * 0.2, 0);

  if (pause_) {
    if (ImGui::Button("Continue", button_size)) {
      pause_ = false;
      tsdf_->SetPause(false);
    }
  } else {
    if (ImGui::Button("Pause", button_size)) {
      pause_ = true;
      tsdf_->SetPause(true);
    }
  }
  if (ImGui::Button("Terminate", button_size)) {
    tsdf_->terminate();
  }
  if (ImGui::Button("Download PC", button_size)) {
    tsdf_->DownloadAll("data.bin");
  }
  if (ImGui::Button("Download mesh", button_size)) {
    tsdf_->DownloadAllMesh("mesh_vertices.bin", "mesh_indices.bin");
  }
  if (ImGui::Button("Follow Camera", button_size)) {
    follow_cam_ = true;
  }
}

void ImageRenderer::DefineGUISlider() {
  // Set push item width to make sliderfloat fill the side menu
  ImGui::PushItemWidth(display_w * 0.2);

  // The `step' parameter pushes virtual camera back and make viewangle wider
  static float step = 0;
  ImGui::Text("Camera Offset");
  ImGui::SliderFloat("##CAM", &step, 0.0f, 3.0f);
  if (follow_cam_) {
    virtual_cam_T_world_ =
        SE3<float>(cam_T_world_.GetR(), cam_T_world_.GetT() + Eigen::Vector3f(0, 0, step));
  }

  // Raycast depth decides the maximum distance ray casting algorithm traverses
  ImGui::Text("Raycast Depth");
  ImGui::SliderFloat("##RAY", &raycast_depth, 4.0f, 20.0f);
}

void ImageRenderer::DefineGUITexture() {
  ImGui::Text("Rendering Texture");
  ImGui::RadioButton("RGB", &render_mode, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Normal", &render_mode, 1);
}

void ImageRenderer::InitializeGUI() {
  // Prep OpenGL window
  // Read current window width and height from OpenGL
  glfwGetFramebufferSize(window_, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);

  // GUI Initiliazation
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_w * 0.2, display_h));
  ImGui::Begin("Menu");
}
