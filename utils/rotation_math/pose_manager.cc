#include "utils/rotation_math/pose_manager.h"

pose_manager::pose_manager() {
  // nothing
}

void pose_manager::register_valid_pose(const int64_t timestamp, const SE3<float> &pose) {
  std::lock_guard<std::mutex> lock(vec_lock);
  timed_pose_tuple new_observation;
  new_observation.first = timestamp;
  new_observation.second = pose;

  timed_pose_vec.push_back(new_observation);
}

void pose_manager::register_valid_pose(const int64_t timestamp, const Eigen::Matrix4d& pose) {
  std::lock_guard<std::mutex> lock(vec_lock);
  timed_pose_tuple new_observation;
  new_observation.first = timestamp;

  /* Translate Eigen::Matrix4d to SE3<float> */
  new_observation.second = SE3<float>(
    Eigen::Matrix<float, 4, 4>(
      pose.cast<float>()
    )
  );

  timed_pose_vec.push_back(new_observation);
}

SE3<float> pose_manager::query_pose(const int64_t timestamp) {
  std::lock_guard<std::mutex> lock(vec_lock);
  if (timed_pose_vec.empty()) {
    // return identity matrix
    return SE3<float>::Identity();
  }
  // find idx
  uint64_t max_lower_idx = get_max_lower_idx(timestamp, 0, (uint64_t)(timed_pose_vec.size()));
  if (max_lower_idx == (uint64_t)(timed_pose_vec.size() - 1)) {
    // last element. Just use it as return
    return timed_pose_vec[max_lower_idx].second;
  } else {
    // not the last element
    int64_t old_timestamp = timed_pose_vec[max_lower_idx].first;
    int64_t new_timestamp = timed_pose_vec[max_lower_idx + 1].first;
    assert(old_timestamp <= timestamp);
    assert(new_timestamp > timestamp);
    // use closet pose right now
    // TODO: use SLERP interpolation to get more accurate pose
    if ((timestamp - old_timestamp) < (new_timestamp - timestamp)) {
      // closer to old timestamp
      return timed_pose_vec[max_lower_idx].second;
    } else {
      // closer to new timestamp
      return timed_pose_vec[max_lower_idx + 1].second;
    }
  }
}

SE3<float> pose_manager::get_latest_pose() {
  std::lock_guard<std::mutex> lock(vec_lock);
  if (timed_pose_vec.empty()) {
    // return identity matrix
    return SE3<float>::Identity();
  }
  size_t latest_idx = (size_t)(timed_pose_vec.size() - 1);
  return timed_pose_vec[latest_idx].second;
}

uint64_t pose_manager::get_max_lower_idx(const int64_t timestamp, uint64_t start_idx,
                                         uint64_t end_idx) {
  if ((end_idx - start_idx) < 42) {
    // brute force
    for (uint64_t i = start_idx; i < end_idx; ++i) {
      if (timed_pose_vec[i].first > timestamp) {
        return i - 1;
      }
    }
    // not terminated at end of vector...
    assert(end_idx == (uint64_t)(timed_pose_vec.size()));
    return end_idx - 1;
  }
  // Recursive case: binary search
  uint64_t mid_idx = (start_idx + end_idx) / 2;
  if (timed_pose_vec[mid_idx].first <= timestamp) {
    return get_max_lower_idx(timestamp, mid_idx, end_idx);
  } else {
    return get_max_lower_idx(timestamp, start_idx, mid_idx + 1);
  }
}
