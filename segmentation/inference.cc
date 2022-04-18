#include "inference.h"

#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

#include "utils/cuda/errors.cuh"
#include "utils/time.hpp"

inference_engine::inference_engine(const std::string& compiled_engine_path, int width, int height) {
  this->width_ = width;
  this->height_ = height;

  this->whole_width_ = ((int)(width / 32) + 1) * 32;
  this->whole_height_ = ((int)(height / 32) + 1) * 32;

  if (compiled_engine_path == "") {
    spdlog::warn("Model path not provided. Not performing segmentation.");
    this->running_ = false;
  } else {
    this->engine_ = torch::jit::load(compiled_engine_path);
    this->engine_.to(torch::kCUDA);
    this->running_ = true;

    spdlog::debug("[SEGM] Model loaded on GPU.");
  }
}

torch::Tensor inference_engine::infer_one(const cv::Mat& rgb_img) {
  std::vector<cv::Mat> ret;

  if (!running_) {
    return torch::zeros({NUM_CLASSES, this->height_, this->width_},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }
  spdlog::debug("[SEGM] Input rgb image rows: {} cols: {}", rgb_img.rows, rgb_img.cols);

  torch::Tensor rgb_img_tensor = mat_to_tensor(rgb_img);
  torch::Tensor rgb_img_tensor_cuda = rgb_img_tensor.to(torch::kCUDA);

  this->input_buffer_.clear();
  this->input_buffer_.push_back(rgb_img_tensor_cuda);

  const auto start_cp = GetTimestamp<std::chrono::milliseconds>();
  torch::Tensor ht_lt_prob_map =
      this->engine_.forward(this->input_buffer_).toTensor().squeeze().detach();
  const auto end_cp = GetTimestamp<std::chrono::milliseconds>();
  spdlog::debug("[SEGM] Engine forward pass took {} ms", end_cp - start_cp);

  return ht_lt_prob_map;
}
