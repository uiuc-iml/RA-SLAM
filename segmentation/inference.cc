#include "inference.h"

#include <spdlog/spdlog.h>

#include <iostream>

#include "utils/time.hpp"

// mat to tensor
torch::Tensor inference_engine::mat_to_tensor(cv::Mat& my_mat) {
  // sizes: (1, C, H, W)
  // normalization
  my_mat.convertTo(my_mat, CV_32FC3, 1.0f / 255.0f);
  // opencv format H*W*C
  auto input_tensor = torch::from_blob(my_mat.data, {1, my_mat.rows, my_mat.cols, 3});
  // pytorch format N*C*H*W
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  return input_tensor;
}

// (CPU) float32 tensor to float32 mat
cv::Mat inference_engine::float_tensor_to_float_mat(const torch::Tensor& my_tensor) {
  torch::Tensor temp_tensor;
  temp_tensor = my_tensor.to(torch::kCPU);
  cv::Mat ret(temp_tensor.sizes()[0], temp_tensor.sizes()[1], CV_32FC1);
  // copy the data from out_tensor to resultImg
  std::memcpy((void*)ret.data, temp_tensor.data_ptr(), sizeof(float) * temp_tensor.numel());
  cv::resize(ret, ret, cv::Size(this->width_, this->height_));
  return ret;
}

// (CPU) float32 tensor to uint8 mat
cv::Mat inference_engine::float_tensor_to_uint8_mat(const torch::Tensor& my_tensor) {
  torch::Tensor temp_tensor;
  temp_tensor = my_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  temp_tensor = temp_tensor.to(torch::kCPU);
  cv::Mat ret(temp_tensor.sizes()[0], temp_tensor.sizes()[1], CV_8UC1);
  // copy the data from out_tensor to resultImg
  std::memcpy((void*)ret.data, temp_tensor.data_ptr(), sizeof(uint8_t) * temp_tensor.numel());
  return ret;
}

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

std::vector<cv::Mat> inference_engine::infer_one(const cv::Mat& rgb_img, bool ret_uint8_flag) {
  std::vector<cv::Mat> ret;

  if (!running_) {
    cv::Mat ret_img = cv::Mat::ones(this->height_, this->width_, CV_32FC1);
    ret.push_back(ret_img);
    ret.push_back(ret_img);
    return ret;
  }
  cv::Mat rescaled_rgb_img;

  spdlog::debug("[SEGM] Input rgb image rows: {} cols: {}", rgb_img.rows, rgb_img.cols);

  // TODO: use multi-scale model that does not require size to be multiples of 32
  cv::resize(rgb_img, rescaled_rgb_img, cv::Size(this->whole_width_, this->whole_height_));

  torch::Tensor rescaled_rgb_img_tensor = mat_to_tensor(rescaled_rgb_img);
  torch::Tensor rescaled_rgb_img_tensor_cuda = rescaled_rgb_img_tensor.to(torch::kCUDA);

  this->input_buffer_.clear();
  this->input_buffer_.push_back(rescaled_rgb_img_tensor_cuda);

  const auto start_cp = GetTimestamp<std::chrono::milliseconds>();
  torch::Tensor ht_lt_prob_map =
      this->engine_.forward(this->input_buffer_).toTensor().squeeze().detach();
  const auto end_cp = GetTimestamp<std::chrono::milliseconds>();
  spdlog::debug("[SEGM] Engine forward pass took {} ms", end_cp - start_cp);

  if (ret_uint8_flag) {
    ret.push_back(float_tensor_to_uint8_mat(ht_lt_prob_map[0]));
    ret.push_back(float_tensor_to_uint8_mat(ht_lt_prob_map[1]));
  } else {
    ret.push_back(float_tensor_to_float_mat(ht_lt_prob_map[0]));
    ret.push_back(float_tensor_to_float_mat(ht_lt_prob_map[1]));
  }

  return ret;
}
