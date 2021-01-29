#include "inference.h"

#include <iostream>

// mat to tensor
torch::Tensor mat_to_tensor(cv::Mat& my_mat) {
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
cv::Mat float_tensor_to_float_mat(const torch::Tensor& my_tensor) {
  torch::Tensor temp_tensor;
  temp_tensor = my_tensor.to(torch::kCPU);
  cv::Mat ret(temp_tensor.sizes()[0], temp_tensor.sizes()[1], CV_32FC1);
  // copy the data from out_tensor to resultImg
  std::memcpy((void*)ret.data, temp_tensor.data_ptr(), sizeof(float) * temp_tensor.numel());
  cv::resize(ret, ret, cv::Size(640, 360));
  return ret;
}

// (CPU) float32 tensor to uint8 mat
cv::Mat float_tensor_to_uint8_mat(const torch::Tensor& my_tensor) {
  torch::Tensor temp_tensor;
  temp_tensor = my_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  temp_tensor = temp_tensor.to(torch::kCPU);
  cv::Mat ret(temp_tensor.sizes()[0], temp_tensor.sizes()[1], CV_8UC1);
  // copy the data from out_tensor to resultImg
  std::memcpy((void*)ret.data, temp_tensor.data_ptr(), sizeof(uint8_t) * temp_tensor.numel());
  return ret;
}

inference_engine::inference_engine(const std::string& compiled_engine_path) {
  this->engine = torch::jit::load(compiled_engine_path);
  this->engine.to(torch::kCUDA);

  std::cout << "Model loaded." << std::endl;
}

std::vector<cv::Mat> inference_engine::infer_one(const cv::Mat& rgb_img, bool ret_uint8_flag) {
  cv::Mat downsized_rgb_img;
  std::vector<cv::Mat> ret;
  cv::resize(rgb_img, downsized_rgb_img, cv::Size(640, 352));

  torch::Tensor downsized_rgb_img_tensor = mat_to_tensor(downsized_rgb_img);
  torch::Tensor downsized_rgb_img_tensor_cuda = downsized_rgb_img_tensor.to(torch::kCUDA);

  this->input_buffer.clear();
  this->input_buffer.push_back(downsized_rgb_img_tensor_cuda);

  torch::Tensor ht_lt_prob_map =
      this->engine.forward(this->input_buffer).toTensor().squeeze().detach();

  if (ret_uint8_flag) {
    ret.push_back(float_tensor_to_uint8_mat(ht_lt_prob_map[0]));
    ret.push_back(float_tensor_to_uint8_mat(ht_lt_prob_map[1]));
  } else {
    ret.push_back(float_tensor_to_float_mat(ht_lt_prob_map[0]));
    ret.push_back(float_tensor_to_float_mat(ht_lt_prob_map[1]));
  }

  return ret;
}
