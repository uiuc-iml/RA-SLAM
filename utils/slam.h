#pragma once

#include "openvslam/config.h"
#include "openvslam/data/bow_vocabulary.h"
#include "openvslam/type.h"
#include "openvslam/system.h"

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>

class slam_system : public openvslam::system {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //! Constructor
  slam_system(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path);

  //! Feed a stereo frame to SLAM system
  //! (Note: Left and Right images must be stereo-rectified)
  unsigned int feed_stereo_images(const cv::Mat& left_img, const cv::Mat& right_img, 
                                  const double timestamp, const cv::Mat& mask = cv::Mat{});

  //! Save the frame trajectory in the specified format
  void save_matched_trajectory(const std::string &path, 
                               const std::vector<unsigned int> &frame_ids) const; 
};

