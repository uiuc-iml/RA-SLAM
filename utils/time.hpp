#pragma once

#include <chrono>
#include <cinttypes>

static const auto start = std::chrono::steady_clock::now();

template <typename UNIT>
inline int64_t get_timestamp() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<UNIT>(now - start).count();
}

template <typename UNIT>
class LocalClock {
 public:
  LocalClock(int64_t local_tick_now) : offset_(get_timestamp<UNIT>() - local_tick_now) {}

  int64_t convert_timestamp(int64_t local_tick) const {
    return local_tick + offset_;
  }

 private:
  const int64_t offset_;
};
