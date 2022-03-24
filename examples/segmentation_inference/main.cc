#include <chrono>
#include <cinttypes>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "segmentation/inference.h"
#include "utils/tensor_helper/libtorch_helper.h"

int main() {
  // Load image for test run
  cv::Mat image_rgb, image_bgr;
  image_bgr = cv::imread("/home/roger/hospital_images/24.jpg");
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  inference_engine my_engine("/home/roger/disinfect-slam/segmentation/ht_lt.pt", image_bgr.cols,
                             image_bgr.rows);

  // Test inference and uint8 conversion
  const auto start = std::chrono::steady_clock::now();
  torch::Tensor ret_prob_map = my_engine.infer_one(image_rgb);
  const auto now = std::chrono::steady_clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  std::cout << "Time elapsed (in milliseconds): " << time_elapsed << std::endl;
  std::cout << "Test image feeded." << std::endl;
  std::cout << "Saving prob maps to current directory." << std::endl;
  cv::imwrite("ht_prob.png", float_tensor_to_uint8_mat(ret_prob_map[0]));
  cv::imwrite("lt_prob.png", float_tensor_to_uint8_mat(ret_prob_map[1]));

  // Benchmark performance
  int num_trials = 1000;
  const auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < num_trials; ++i) {
    auto ret_prob_map = my_engine.infer_one(image_rgb);
  }
  const auto loop_end = std::chrono::steady_clock::now();
  auto loop_total =
      std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
  std::cout << "Loop total time (in milliseconds): " << loop_total << std::endl;
  std::cout << "Inference time per image (in milliseconds): " << ((uint32_t)loop_total / num_trials)
            << std::endl;
  return 0;
}
