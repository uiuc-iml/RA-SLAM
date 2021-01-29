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
   */
  ImageRenderer(const std::string& name, const std::shared_ptr<SLAMSystem>& slam,
                const std::shared_ptr<TSDFSystem>& tsdf, const std::string& config_file_path);

 protected:
  void DispatchInput() override;

  void Render() override;

  void RenderExit() override;

 private:
  bool follow_cam_ = true;
  GLImage8UC4 tsdf_normal_;
  std::shared_ptr<SLAMSystem> slam_;
  std::shared_ptr<TSDFSystem> tsdf_;
  std::shared_ptr<openvslam::publish::map_publisher> map_publisher_;
  SE3<float> cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_T_world_old_ = SE3<float>::Identity();
  const YAML::Node config_;
  const CameraParams virtual_cam_;
};
