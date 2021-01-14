#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cinttypes>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "segmentation/inference.h"

int main() {
    inference_engine my_engine("/home/roger/disinfect-slam/segmentation/ht_lt.pt");
    // Load image for test run
    cv::Mat image_rgb, image_bgr;
    image_bgr = cv::imread("/home/roger/hospital_images/24.jpg");
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

    my_engine.infer_one(image_rgb);
    return 0;
}