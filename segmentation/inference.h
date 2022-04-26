#pragma once

#include <torch/script.h>

#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "utils/tensor_helper/libtorch_helper.h"
#include "utils/tsdf/voxel_types.cuh"  // for NUM_CLASSES

class inference_engine {
 public:
  inference_engine(const std::string& compiled_engine_path, int width, int height);

  torch::Tensor infer_one(const cv::Mat& rgb_img);

 private:
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
