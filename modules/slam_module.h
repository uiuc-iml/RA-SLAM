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

class SLAMSystem : public openvslam::system {
 public:
  //! Constructor
  SLAMSystem(const std::shared_ptr<openvslam::config> &cfg,
              const std::string &vocab_file_path);

  //! Feed a stereo frame to SLAM system
  //! (Note: Left and Right images must be stereo-rectified)
  unsigned int feed_stereo_images(const cv::Mat &img_left, const cv::Mat &img_right,
                                  const double timestamp, const cv::Mat &mask = cv::Mat{});

  unsigned int feed_rgbd_images(const cv::Mat &img_rgb, const cv::Mat &img_depth,
                                const double timestamp, const cv::Mat &mask = cv::Mat{});

  //! Save the frame trajectory in the specified format
  void save_matched_trajectory(const std::string &path,
                               const std::vector<unsigned int> &frame_ids) const;
};

