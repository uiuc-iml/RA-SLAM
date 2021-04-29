#pragma once

#include <openvslam/publish/map_publisher.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "modules/slam_module.h"
#include "modules/tsdf_module.h"
#include "utils/gl/renderer_base.h"

class ImageRenderer : public RendererBase {
 public:
  /**
   * @brief rendering module constructor
   *
   * @param name              name of the renderer window
   * @param slam              shared pointer to a slam system
   * @param tsdf              shared pointer to a TSDF system
   * @param config_file_path  configuration file path
   *
   */
  ImageRenderer(const std::string& name, std::function<SE3<float>()> get_latest_pose_func,
                const std::shared_ptr<TSDFSystem>& tsdf, const CameraIntrinsics<float> intrinsics);

 protected:
  /**
   * @brief Fetch mouse input from ImGUIIO object and directs them to right handler.
   *        In paricular, three types of events are taken care of:
   *            - Mouse wheel scrolling
   *            - Left click & dragging
   *            - Right click & dragging
   *
   *        See the following three functions for details on handling these events
   */
  void DispatchInput() override;

  /**
   * @brief Handle mouse wheel scrolling event (move along z-axis of virtual camera)
   */
  void HandleMouseScrollWheel(const ImGuiIO& io);

  /**
   * @brief Handle mouse left click dragging event (rotate virtual camera)
   */
  void HandleMouseLeftClick(const ImGuiIO& io);

  /**
   * @brief Handle mouse middle click dragging event (translate camera plane)
   */
  void HandleMouseMiddleClick(const ImGuiIO& io);

  /**
   * @brief Main function that is repeatedly called by the RendererBase. It does the
   *        following steps:
   *            1. Fetch some useful info (e.g., window closed? window size) from OpenGL
   *            2. Use these infos to call ImGUI routine to render relevant GUI
   *            3. Call TSDF ray casting routine to render OpenGL image with useful pixels
   *            4. Detect if termination is requested
   */
  void Render() override;

  /**
   * @brief Helper function that terminates rendering loop
   */
  void RenderExit() override;

  /**
   * @brief Routine to fetch info from OpenGL and initialize ImGUI.
   *
   * @NOTE: need to be called before any other GUI definition!
   */
  void InitializeGUI();

  /**
   * @brief Define and render some basic buttons in GUI
   */
  void DefineGUIButton();

  /**
   * @brief Define and render some sliders that control several useful parameters
   */
  void DefineGUISlider();

  /**
   * @brief Define switches in the GUI, which allow user to specify types of texture to render
   */
  void DefineGUITexture();

 private:
  bool follow_cam_ = true;
  bool pause_ = false;
  int render_mode = 1;
  float raycast_depth = 10;
  int display_w, display_h;
  GLImage8UC4 tsdf_normal_;
  GLImage8UC4 tsdf_rgba_;
  std::shared_ptr<SLAMSystem> slam_;
  std::shared_ptr<TSDFSystem> tsdf_;
  SE3<float> cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_old_ = SE3<float>::Identity();
  const CameraParams virtual_cam_;
  std::function<SE3<float>()> get_latest_pose_func_;
};
