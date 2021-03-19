#include <iostream>

#include "utils/scannet_sens_reader/scannet_sens_reader.h"

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
  Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  std::cout << "R: " << extrinsics.GetR().normalized().toRotationMatrix() << std::endl;
  std::cout << "t: " << extrinsics.GetT() << std::endl;
  // print depth factor
  float depth_factor = my_reader.get_depth_map_factor();
  std::cout << "depth factor: " << depth_factor << std::endl;
  // print available frame size
  int stream_size = my_reader.get_size();
  std::cout << "Stream size: " << stream_size << std::endl;
  // save RGB image
  cv::Mat rgb_img;
  my_reader.get_color_frame_by_id(&rgb_img, 0);
  cv::imwrite("rgb_img_0.jpg", rgb_img);
  // save depth image
  cv::Mat depth_img;
  my_reader.get_depth_frame_by_id(&depth_img, 0);
  cv::imwrite("depth_img_0.png", depth_img);
}