#pragma once

#include <spdlog/spdlog.h>

#include <mutex>
#include <thread>

/**
 * @brief asynchronous data logger class
 *
 * @tparam T  type of data to be logged
 */
template <class T>
class DataLogger {
 public:
  /**
   * @brief start logging thread
   */
  DataLogger() : log_thread_(&DataLogger::Run, this) {}

  /**
   * @brief stop logging
   */
  virtual ~DataLogger() {
    {
      std::lock_guard<std::mutex> lock(mtx_terminate_);
      terminate_is_requested_ = true;
    }
    log_thread_.join();
  }

  /**
   * @brief log data to disk
   *
   * @param data data to be logged
   */
  void LogData(const T& data) {
    std::lock_guard<std::mutex> lock(mtx_data_);
    if (data_available_ == true) {
      spdlog::warn("Logger cannot catch up, data is being dropped");
    }
    data_[write_idx_] = T(data);  // explicitly calls T's copy constructor
    data_available_ = true;
  }

 protected:
  /**
   * @brief child class should implement this to serialize data to disk
   *
   * @param data data to be serialized and saved
   */
  virtual void SaveData(const T& data) = 0;

 private:
  std::mutex mtx_data_;
  int write_idx_ = 0;
  bool data_available_ = false;
  T data_[2];

  std::thread log_thread_;
  std::mutex mtx_terminate_;
  bool terminate_is_requested_ = false;

  void Run() {
    while (true) {
      // check terminate request
      {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        if (terminate_is_requested_) break;
      }
      // swap read / write buffer
      {
        std::lock_guard<std::mutex> lock(mtx_data_);
        if (!data_available_) continue;
        data_available_ = false;
        write_idx_ = 1 - write_idx_;
      }
      SaveData(data_[1 - write_idx_]);
    }
  }
};
