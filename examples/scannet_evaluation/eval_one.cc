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
#include <vector>

#include "segmentation/inference.h"
#include "third_party/indicators.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/cuda/vector.cuh"
#include "utils/gl/image.h"
#include "utils/offline_data_provider/scannet_sens_reader.h"
#include "utils/time.hpp"
#include "utils/tsdf/voxel_tsdf.cuh"

class DatasetEvaluator {
 public:
  DatasetEvaluator(const std::string& segm_model_path, const std::string& sens_path,
                   const std::string& tsdf_path)
      : segm_(segm_model_path, 640, 480),
        sens_reader_(sens_path),
        tsdf_(0.01, 0.06),
        intrinsics_(sens_reader_.get_camera_intrinsics()),
        depth_scale_(sens_reader_.get_depth_map_factor()),
        tsdf_path_(tsdf_path) {
    spdlog::debug("[RGBD Intrinsics] fx: {} fy: {} cx: {} cy: {}", intrinsics_.fx, intrinsics_.fy,
                  intrinsics_.cx, intrinsics_.cy);
  }

  void run_all() {
    cv::Mat img_rgb, img_depth, img_ht, img_lt;
    int cur_percentage = 0;
    indicators::ProgressBar bar_{indicators::option::BarWidth{50},
                                 indicators::option::Start{" ["},
                                 indicators::option::Fill{"█"},
                                 indicators::option::Lead{"█"},
                                 indicators::option::Remainder{"-"},
                                 indicators::option::End{"]"},
                                 indicators::option::PrefixText{"Evaluating ScanNet scene"},
                                 indicators::option::ForegroundColor{indicators::Color::yellow},
                                 indicators::option::ShowElapsedTime{true},
                                 indicators::option::ShowRemainingTime{true}};
    spdlog::info("Starting evaluation! Total frame count: {}", sens_reader_.get_size());
    for (int frame_idx = 0; frame_idx < sens_reader_.get_size(); ++frame_idx) {
      if (((int)(frame_idx * 1.0 / sens_reader_.get_size() * 100)) > cur_percentage) {
        cur_percentage = (int)(frame_idx * 1.0 / sens_reader_.get_size() * 100);
        bar_.tick();
      }
      // 1. read RGB and depth images
      const auto st_img = GetTimestamp<std::chrono::milliseconds>();
      sens_reader_.get_color_frame_by_id(&img_rgb, frame_idx);
      sens_reader_.get_depth_frame_by_id(&img_depth, frame_idx);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / depth_scale_);
      const auto end_img = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("Image IO takes {} ms", end_img - st_img);
      // 2. infer RGB image
      const auto st_segm = GetTimestamp<std::chrono::milliseconds>();
      std::vector<cv::Mat> prob_map = segm_.infer_one(img_rgb, false);
      const auto end_segm = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("Segmentation takes {} ms", end_segm - st_segm);
      // 3. TSDF integration
      const auto st = GetTimestamp<std::chrono::milliseconds>();
      SE3<float> cam_T_world = sens_reader_.get_camera_pose_by_id(frame_idx);
      tsdf_.Integrate(img_rgb, img_depth, prob_map[0], prob_map[1], 4, intrinsics_, cam_T_world);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      const auto end = GetTimestamp<std::chrono::milliseconds>();
      spdlog::debug("Integration takes {} ms", end - st);
    }
    // all images visited, terminate
    spdlog::info("Evaluation completed! Extracting TSDF now...");
    const auto voxel_pos_prob = tsdf_.GatherValidSemantic();
    spdlog::info("Visible TSDF blocks count: {}", voxel_pos_prob.size());
    std::ofstream fout(tsdf_path_, std::ios::out | std::ios::binary);
    fout.write((char*)voxel_pos_prob.data(), voxel_pos_prob.size() * sizeof(VoxelSpatialTSDFSEGM));
    fout.close();
  }

 private:
  TSDFGrid tsdf_;
  inference_engine segm_;
  scannet_sens_reader sens_reader_;
  const CameraIntrinsics<float> intrinsics_;
  const float depth_scale_;
  std::string tsdf_path_;
};

int main(int argc, char* argv[]) {
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto model = op.add<popl::Value<std::string>>("", "model", "path PyTorch JIT traced model");
  auto sens =
      op.add<popl::Value<std::string>>("", "sens", "path to scannet sensor stream .sens file");
  auto tsdf_path =
      op.add<popl::Value<std::string>>("", "tsdf", "path to dump semantic TSDF reconstruction");

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

  if (!model->is_set() || !sens->is_set() || !tsdf_path->is_set()) {
    spdlog::error("Invalid arguments");
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  DatasetEvaluator evaluator(model->value(), sens->value(), tsdf_path->value());
  evaluator.run_all();

  return EXIT_SUCCESS;
}
