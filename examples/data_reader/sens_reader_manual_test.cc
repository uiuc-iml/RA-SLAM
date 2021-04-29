#include <iostream>

#include "segmentation/inference.h"
#include "utils/offline_data_provider/scannet_sens_reader.h"

int main(int argc, char* argv[]) {
  std::string filename = "scene0001_00.sens";
  if (argc >= 2) {
    filename = std::string(argv[1]);
  } else {
    std::cout << "Please provide path to the test .sens file" << std::endl;
  }

  scannet_sens_reader my_reader(filename);

  // print intrinsics
  CameraIntrinsics<float> intrinsics = my_reader.get_camera_intrinsics();
  std::cout << "fx: " << intrinsics.fx << std::endl;
  std::cout << "fy: " << intrinsics.fy << std::endl;
  std::cout << "cx: " << intrinsics.cx << std::endl;
  std::cout << "cy: " << intrinsics.cy << std::endl;
  // print extrinsics
  SE3<float> extrinsics = my_reader.get_camera_extrinsics();
  std::cout << "Extrinsics: " << std::endl;
  std::cout << "R: " << extrinsics.GetR().normalized().toRotationMatrix() << std::endl;
  std::cout << "t: " << extrinsics.GetT() << std::endl;
  // print depth factor
  float depth_factor = my_reader.get_depth_map_factor();
  std::cout << "depth factor: " << depth_factor << std::endl;
  // print available frame size
  int stream_size = my_reader.get_size();
  std::cout << "Stream size: " << stream_size << std::endl;
  // save RGB image at frame 0
  cv::Mat rgb_img;
  inference_engine my_engine("/home/roger/disinfect-slam/segmentation/ht_lt.pt",
                             my_reader.get_width(), my_reader.get_height());
  my_reader.get_color_frame_by_id(&rgb_img, 243);
  std::cout << "Rgb image cols: " << rgb_img.cols << std::endl;
  std::cout << "Rgb image rows: " << rgb_img.rows << std::endl;
  std::vector<cv::Mat> ret_prob_map = my_engine.infer_one(rgb_img, false);
  cv::Mat vis_ht, vis_lt;
  ret_prob_map[0].convertTo(vis_ht, CV_8UC1,
                            255);  // scale range from 0-1 to 0-255
  ret_prob_map[1].convertTo(vis_lt, CV_8UC1,
                            255);  // scale range from 0-1 to 0-255
  cv::imwrite("float_ht_prob.png", vis_ht);
  cv::imwrite("float_lt_prob.png", vis_lt);
  cv::imwrite("rgb_img_0.jpg", rgb_img);
  // save depth image at frame 0
  cv::Mat depth_img;
  my_reader.get_depth_frame_by_id(&depth_img, 243);
  std::cout << "Depth image cols: " << depth_img.cols << std::endl;
  std::cout << "Depth image rows: " << depth_img.rows << std::endl;
  cv::imwrite("depth_img_0.png", depth_img);
  // print camera pose at frame 0
  SE3<float> pose = my_reader.get_camera_pose_by_id(0);
  std::cout << "Pose: " << std::endl;
  std::cout << "R: " << pose.GetR().normalized().toRotationMatrix() << std::endl;
  std::cout << "t: " << pose.GetT() << std::endl;
}