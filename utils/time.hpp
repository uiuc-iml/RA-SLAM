#pragma once

#include <chrono>
#include <cinttypes>

static const auto start = std::chrono::steady_clock::now();

// get timestamp from steady clock
// this is preferred
template <typename UNIT>
inline int64_t GetTimestamp() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<UNIT>(now - start).count();
}

// get timestamp from unsteady system clock
// have to have this function because librealsense is using system clock...
template <typename UNIT>
inline int64_t GetSystemTimestamp() {
  const auto tp = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<UNIT>(tp).count();
}

template <typename UNIT>
class LocalClock {
 public:
  LocalClock(int64_t local_tick_now) : offset_(GetTimestamp<UNIT>() - local_tick_now) {}

  int64_t convert_timestamp(int64_t local_tick) const {
    return local_tick + offset_;
  }

 private:
  const int64_t offset_;
};
