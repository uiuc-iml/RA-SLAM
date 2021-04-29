#pragma once

#include <torch/script.h>

#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class inference_engine {
 public:
  inference_engine(const std::string& compiled_engine_path, int width, int height);

  // ret[0]: ht map
  // ret[1]: lt map
  std::vector<cv::Mat> infer_one(const cv::Mat& rgb_img, bool ret_uint8_flag);

 private:
  torch::Tensor mat_to_tensor(cv::Mat& my_mat);

  cv::Mat float_tensor_to_float_mat(const torch::Tensor& my_tensor);

  cv::Mat float_tensor_to_uint8_mat(const torch::Tensor& my_tensor);

  torch::jit::script::Module engine_;
  std::vector<torch::jit::IValue> input_buffer_;

  /* Width and height are input (and desired output) image dimensions */
  int width_;
  int height_;

  /* Some segmentation model requires resolution to be multiples of 32 */
  /* whole_width and whole_height pad this resolution */
  int whole_width_;
  int whole_height_;

  /* if running_ is False, then no segmentation will be performed. */
  bool running_;
};
