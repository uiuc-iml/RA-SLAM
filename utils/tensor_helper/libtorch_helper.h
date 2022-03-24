#pragma once

#include <torch/script.h>

#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat float_tensor_to_uint8_mat(const torch::Tensor& my_tensor);

torch::Tensor mat_to_tensor(const cv::Mat& my_mat);

cv::Mat float_tensor_to_float_mat(const torch::Tensor& my_tensor);
