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

    // Test inference and uint8 conversion
    const auto start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> ret_prob_map = my_engine.infer_one(image_rgb, true);
    const auto now = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << "Time elapsed (in milliseconds): " << time_elapsed << std::endl;
    std::cout << "Test image feeded." << std::endl;
    std::cout << "Saving prob maps to current directory." << std::endl;
    cv::imwrite("ht_prob.png", ret_prob_map[0]);
    cv::imwrite("lt_prob.png", ret_prob_map[1]);

    // Test inference and float conversion
    const auto n_start = std::chrono::steady_clock::now();
    ret_prob_map.clear();
    ret_prob_map = my_engine.infer_one(image_rgb, false);
    const auto n_now = std::chrono::steady_clock::now();
    auto n_time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(n_now - n_start).count();
    std::cout << "Time elapsed (in milliseconds): " << n_time_elapsed << std::endl;
    std::cout << "Test image feeded." << std::endl;
    std::cout << "Saving prob maps to current directory." << std::endl;
    cv::Mat vis_ht, vis_lt;
    ret_prob_map[0].convertTo(vis_ht, CV_8UC1, 255); // scale range from 0-1 to 0-255
    ret_prob_map[1].convertTo(vis_lt, CV_8UC1, 255); // scale range from 0-1 to 0-255
    cv::imwrite("float_ht_prob.png", vis_ht);
    cv::imwrite("float_lt_prob.png", vis_lt);

    // Benchmark performance
    int num_trials = 1000;
    const auto loop_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_trials; ++i) {
        ret_prob_map.clear();
        ret_prob_map = my_engine.infer_one(image_rgb, true);
    }
    const auto loop_end = std::chrono::steady_clock::now();
    auto loop_total = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
    std::cout << "Loop total time (in milliseconds): " << loop_total << std::endl;
    std::cout << "Inference time per image (in milliseconds): " << ((uint32_t)loop_total / num_trials) << std::endl;
    return 0;
}