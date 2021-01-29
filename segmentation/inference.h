#pragma once

#include <torch/script.h>

#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class inference_engine {
 public:
  inference_engine(const std::string& compiled_engine_path);

  // ret[0]: ht map
  // ret[1]: lt map
  std::vector<cv::Mat> infer_one(const cv::Mat& rgb_img, bool ret_uint8_flag);

 private:
  torch::jit::script::Module engine;
  std::vector<torch::jit::IValue> input_buffer;
};
