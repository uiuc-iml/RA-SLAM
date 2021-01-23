#pragma once

#include <iostream>
#include <string>

#include <openvslam/system.h>
#include <openvslam/publish/map_publisher.h>

#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "modules/slam_module.h"
#include "modules/tsdf_module.h"
#include "utils/gl/renderer_base.h"
#include "utils/time.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/rotation_math/pose_manager.h"
#include "utils/config_reader.hpp"

class ImageRenderer : public RendererBase {
 public:
  ImageRenderer(const std::string &name,
                const std::shared_ptr<SLAMSystem> &slam,
                const std::shared_ptr<TSDFSystem> &tsdf,
                const std::string &config_file_path);

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
  SE3<float> cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_ = SE3<float>::Identity();
  SE3<float> virtual_cam_P_world_old_ = SE3<float>::Identity();
  const YAML::Node config_;
  const CameraParams virtual_cam_;
};