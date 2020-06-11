#include "slam.h"

#include <iostream>
#include <iomanip>
#include <spdlog/spdlog.h>

#include "openvslam/data/keyframe.h"
#include "openvslam/data/map_database.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "openvslam/tracking_module.h"

using namespace openvslam;

slam_system::slam_system(const std::shared_ptr<config>& cfg, 
                         const std::string& vocab_file_path)
    : system(cfg, vocab_file_path) {}

void slam_system::save_matched_trajectory(const std::string &path,
                                          const std::vector<unsigned int> &frame_ids) const {
  pause_other_threads();

  const std::unordered_set<unsigned int> frame_ids_set(frame_ids.begin(), frame_ids.end());

  std::ofstream fout(path, std::ios::out);
  if (!fout.is_open()) {
    spdlog::critical("cannot create a file at {}", path);
    throw std::runtime_error("cannot create a file at " + path);
  }
  fout << std::setprecision(9);

  std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
  /* xTy: rigid body transformation from x frame to y frame
   * c: camera
   * w: world
   * k: keyframe */
  const auto frame_stats = map_db_->get_frame_statistics();
  const auto num_valid_frames = frame_stats.get_num_valid_frames();
  const auto reference_keyframes = frame_stats.get_reference_keyframes();
  const auto cam_poses_cTk = frame_stats.get_relative_cam_poses();
  const auto is_lost_frames = frame_stats.get_lost_frames();
  const auto rk_iter_end = reference_keyframes.end();
  const auto rc_iter_end = cam_poses_cTk.end();
  auto rk_iter = reference_keyframes.begin();
  auto rc_iter = cam_poses_cTk.begin();
  for (;rk_iter != rk_iter_end && rc_iter != rc_iter_end; ++rk_iter, ++rc_iter) {
    const auto frame_id = rk_iter->first;
    if (is_lost_frames.at(frame_id) || frame_ids_set.find(frame_id) == frame_ids_set.end())
      continue;
    const auto ref_keyframe = rk_iter->second;
    const Mat44_t cam_pose_kTw = ref_keyframe->get_cam_pose();
    const Mat44_t cam_pose_cTk = rc_iter->second;
    const Mat44_t cam_pose_cTw = cam_pose_cTk * cam_pose_kTw;
    const Mat44_t cam_pose_wTc = cam_pose_cTw.inverse();

    fout << frame_id << " ";
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 4; ++j)
        fout << cam_pose_wTc(i, j) << " ";
    fout << std::endl;
  }

  if (rk_iter != rk_iter_end || rc_iter != rc_iter_end)
    spdlog::error("sizes of frame statistics are not matched");

  resume_other_threads();
}

unsigned int slam_system::feed_stereo_images(
    const cv::Mat& left_img, const cv::Mat& right_img, 
    const double timestamp, const cv::Mat& mask) {
  assert(camera_->setup_type_ == camera::setup_type_t::Stereo);

  check_reset_request();

  const Mat44_t cam_pose_cw = tracker_->track_stereo_image(left_img, right_img, timestamp, mask);

  frame_publisher_->update(tracker_);
  if (tracker_->tracking_state_ == tracker_state_t::Tracking) {
    map_publisher_->set_current_cam_pose(cam_pose_cw);
  }

  return tracker_->curr_frm_->id_;
}

