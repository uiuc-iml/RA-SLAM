#include "zed_native.h"

#include <opencv2/opencv.hpp>

ZEDNative::ZEDNative(const openvslam::config &cfg) 
    : rectifier_(cfg.camera_, cfg.yaml_node_), cam_model_(cfg.camera_) {
}
