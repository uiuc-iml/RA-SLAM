#pragma once

#include "openvslam/camera/base.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"

class ZEDNative {
 public:
  ZEDNative(const openvslam::config &cfg);
  ~ZEDNative();

 private:
  void capture_thread();

  const openvslam::camera::base *cam_model_;
  const openvslam::util::stereo_rectifier rectifier_;
};
