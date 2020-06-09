#include <chrono>

#include "time.h"

static const auto start = std::chrono::steady_clock::now();

float get_timestamp_sec_f() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
}

float get_timestamp_milli_f() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

float get_timestamp_micro_f() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
}

unsigned int get_timestamp_sec_u() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
}

unsigned int get_timestamp_milli_u() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

unsigned int get_timestamp_micro_u() {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
}
