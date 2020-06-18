#pragma once

#include <mutex>
#include <thread>

#include <spdlog/spdlog.h>

template <class T>
class DataLogger {
 public:
  DataLogger() : log_thread_(&DataLogger::run, this) {}

  virtual ~DataLogger() {
    {
      std::lock_guard<std::mutex> lock(mtx_terminate_);
      terminate_is_requested_ = true;
    }
    log_thread_.join();
  }

  void log_data(const T &data) {
    std::lock_guard<std::mutex> lock(mtx_data_);
    if (data_available_ == true) {
      spdlog::warn("Logger cannot catch up, data is being dropped");
    }
    data_[write_idx_] = T(data); // explicitly calls T's copy constructor
    data_available_ = true;
  }

 protected:
  virtual void save_data(const T &data) = 0;

 private:
  std::mutex mtx_data_;
  int write_idx_ = 0;
  bool data_available_ = false;
  T data_[2];

  std::thread log_thread_;
  std::mutex mtx_terminate_;
  bool terminate_is_requested_ = false;

  void run() {
    while (true) {
      // check terminate request
      {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        if (terminate_is_requested_)
          break;
      }
      // swap read / write buffer
      {
        std::lock_guard<std::mutex> lock(mtx_data_);
        if (!data_available_)
          continue;
        data_available_ = false;
        write_idx_ = 1 - write_idx_;
      }
      save_data(data_[1 - write_idx_]);
    }
  }
};
