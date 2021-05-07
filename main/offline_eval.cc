#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <popl.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "modules/renderer_module.h"
#include "modules/tsdf_module.h"
#include "segmentation/inference.h"
#include "utils/cuda/errors.cuh"
#include "utils/cuda/vector.cuh"
#include "utils/offline_data_provider/folder_reader.h"
#include "utils/offline_data_provider/offline_data_provider.h"
#include "utils/offline_data_provider/rosbag_reader.h"
#include "utils/offline_data_provider/scannet_sens_reader.h"
#include "utils/rotation_math/pose_manager.h"
#include "utils/time.hpp"

/* Util implementation from https://stackoverflow.com/questions/874134/ */
bool str_ends_with(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void run(const std::string& segm_model_path, const std::string& data_path, bool rendering_flag) {
  /* Use abstract interface to supply data */
  std::unique_ptr<offline_data_provider> my_datareader;
  if (str_ends_with(data_path, ".sens")) {
    my_datareader = std::make_unique<scannet_sens_reader>(data_path);
  } else if (str_ends_with(data_path, ".bag")) {
    my_datareader = std::make_unique<rosbag_reader>(data_path);
  } else {
    my_datareader = std::make_unique<folder_reader>(data_path);
  }

  /* Initialize various modules */
  auto my_tsdf = std::make_shared<TSDFSystem>(0.01, 0.06, 6, my_datareader->get_camera_intrinsics(),
                                              my_datareader->get_camera_extrinsics());
  pose_manager camera_pose_manager;
  ImageRenderer my_renderer("tsdf", std::bind(&pose_manager::get_latest_pose, &camera_pose_manager),
                            my_tsdf, my_datareader->get_camera_intrinsics());
  inference_engine my_seg(segm_model_path, my_datareader->get_width(), my_datareader->get_height());
  cv::Mat img_rgb, img_depth, img_ht, img_lt;

  std::thread t_tsdf([&]() {
    spdlog::info("Init. Stream size: {}", my_datareader->get_size());

    for (int frame_idx = 0; frame_idx < my_datareader->get_size(); frame_idx++) {
      if (my_tsdf->is_terminated()) break;
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      SE3<float> cam_T_world = my_datareader->get_camera_pose_by_id(frame_idx);
      camera_pose_manager.register_valid_pose((int64_t)(frame_idx), cam_T_world);
      my_datareader->get_color_frame_by_id(&img_rgb, frame_idx);
      my_datareader->get_depth_frame_by_id(&img_depth, frame_idx);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / my_datareader->get_depth_map_factor());
      // use seg engine to get ht/lt img
      cv::imshow("bgr", img_rgb);
      std::vector<cv::Mat> prob_map = my_seg.infer_one(img_rgb, false);
      my_tsdf->Integrate(cam_T_world, img_rgb, img_depth, prob_map[0], prob_map[1]);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / 4);
      cv::imshow("depth", img_depth);
      cv::waitKey(1);
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("[OFFLINE EVAL] Processing frame {} took {} ms", frame_idx, end - st);
    }

    if (!my_tsdf->is_terminated()) my_tsdf->terminate();
  });

  my_renderer.Run();
  if (t_tsdf.joinable()) t_tsdf.join();
}

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto model = op.add<popl::Value<std::string>>("", "model", "path PyTorch JIT traced model");
  auto sens =
      op.add<popl::Value<std::string>>("", "sens", "path to scannet sensor stream .sens file");

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  try {
    op.parse(argc, argv);
  } catch (const std::exception& e) {
    spdlog::error(e.what());
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  if (debug_mode->is_set())
    spdlog::set_level(spdlog::level::debug);
  else
    spdlog::set_level(spdlog::level::info);

  if (!sens->is_set()) {
    spdlog::error("Invalid arguments");
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  run(model->value(), sens->value(), true);

  return EXIT_SUCCESS;
}