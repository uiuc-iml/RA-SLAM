#pragma once

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>
#include <mutex>
#include <assert.h>

#include "utils/cuda/lie_group.cuh"

using timed_pose_tuple = std::pair<int64_t, SE3<float>>;

class pose_manager {
  public:
    pose_manager();

    void register_valid_pose(const int64_t timestamp, const SE3<float> pose);

    SE3<float> query_pose(const int64_t timestamp);

  private:
    // get index to the maximum timestamp that is smaller than the given timestamp.
    uint64_t get_max_lower_idx(const int64_t timestamp, uint64_t start_idx, uint64_t end_idx);

    std::vector<timed_pose_tuple> timed_pose_vec;

    std::mutex vec_lock;
};