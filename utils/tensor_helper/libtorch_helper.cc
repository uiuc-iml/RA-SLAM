#include "libtorch_helper.h"

cv::Mat float_tensor_to_uint8_mat(const torch::Tensor& my_tensor) {
  torch::Tensor temp_tensor;
  temp_tensor = my_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  temp_tensor = temp_tensor.to(torch::kCPU);
  cv::Mat ret(temp_tensor.sizes()[0], temp_tensor.sizes()[1], CV_8UC1);
  // copy the data from out_tensor to resultImg
  std::memcpy((void*)ret.data, temp_tensor.data_ptr(), sizeof(uint8_t) * temp_tensor.numel());
  return ret;
}

// mat to tensor
torch::Tensor mat_to_tensor(const cv::Mat& my_mat) {\
  cv::Mat float_mat;
  // sizes: (1, C, H, W)
  // normalization
  my_mat.convertTo(float_mat, CV_32FC3, 1.0f / 255.0f);
  // opencv format H*W*C
  auto input_tensor = torch::from_blob(float_mat.data, {1, float_mat.rows, float_mat.cols, 3}).clone();
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
  return ret;
}