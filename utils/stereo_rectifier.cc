#include "stereo_rectifier.h"

#include <spdlog/spdlog.h>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

stereo_rectifier::stereo_rectifier(const YAML::Node& yaml_node) {
    spdlog::debug("CONSTRUCT: stereo_rectifier");
    // set image size
    const cv::Size img_size(yaml_node["Camera.cols"].as<unsigned int>(),
                            yaml_node["Camera.rows"].as<unsigned int>());
    // set camera matrices
    const double fx_l = yaml_node["Calibration.left.fx"].as<double>();
    const double fy_l = yaml_node["Calibration.left.fy"].as<double>();
    const double cx_l = yaml_node["Calibration.left.cx"].as<double>();
    const double cy_l = yaml_node["Calibration.left.cy"].as<double>();
    const auto dist_l = yaml_node["Calibration.left.distortion"].as<std::vector<double>>();

    const double fx_r = yaml_node["Calibration.right.fx"].as<double>();
    const double fy_r = yaml_node["Calibration.right.fy"].as<double>();
    const double cx_r = yaml_node["Calibration.right.cx"].as<double>();
    const double cy_r = yaml_node["Calibration.right.cy"].as<double>();
    const auto dist_r = yaml_node["Calibration.right.distortion"].as<std::vector<double>>();

    const auto K_l = get_intrinsics(fx_l, fy_l, cx_l, cy_l);
    const auto K_r = get_intrinsics(fx_r, fy_r, cx_r, cy_r);
    const auto D_l = parse_vector_as_mat(dist_l);
    const auto D_r = parse_vector_as_mat(dist_r);
    const auto R_vec = parse_vector_as_mat(
        yaml_node["Calibration.rotation"].as<std::vector<double>>());
    const auto T = parse_vector_as_mat(
        yaml_node["Calibration.translation"].as<std::vector<double>>()).t();
    const auto R = rotation_vec2mat(R_vec);

    cv::Mat R_l, R_r, P_l;

    // calculate stereo rectification params
    cv::stereoRectify(K_l, D_l, K_r, D_r, img_size, R, T, R_l, R_r, P_l, 
                      cam_rect_matrix_, reproj_mat_, cv::CALIB_ZERO_DISPARITY, 0, img_size);

    // log camera calibration parameters
    std::stringstream ss;
    ss << "[Camera Calibation Parameteres]" << std::endl;
    ss << "Camera Intrinsics Left: \n"      << K_l              << std::endl
       << "Camera Intrinsics Right: \n"     << K_r              << std::endl
       << "Camera Distortion Left: \n"      << D_l              << std::endl
       << "Camera Distortion Right: \n"     << D_r              << std::endl
       << "Camera Right Rotation: \n"       << R                << std::endl
       << "Camera Right Translation \n"     << T                << std::endl
       << "Camera Intrinsics Rectified: \n" << cam_rect_matrix_ << std::endl;
    spdlog::debug(ss.str());

    // set distortion parameters depending on the camera model
    // create undistortion maps
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l, img_size, CV_32F, 
        undist_map_x_l_, undist_map_y_l_);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, cam_rect_matrix_, img_size, CV_32F, 
        undist_map_x_r_, undist_map_y_r_);
}

stereo_rectifier::~stereo_rectifier() {
    spdlog::debug("DESTRUCT: stereo_rectifier");
}

void stereo_rectifier::rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r,
                               cv::Mat& out_img_l, cv::Mat& out_img_r) const {
    cv::remap(in_img_l, out_img_l, undist_map_x_l_, undist_map_y_l_, cv::INTER_LINEAR);
    cv::remap(in_img_r, out_img_r, undist_map_x_r_, undist_map_y_r_, cv::INTER_LINEAR);
}

const cv::Mat stereo_rectifier::get_rectified_intrinsics() const {
  return cam_rect_matrix_;
}

cv::Mat stereo_rectifier::get_intrinsics(double fx, double fy, double cx, double cy) const {
  cv::Mat intrinsics = cv::Mat::zeros(3, 3, CV_64FC1);

  intrinsics.at<double>(0, 0) = fx;
  intrinsics.at<double>(0, 2) = cx;
  intrinsics.at<double>(1, 1) = fy;
  intrinsics.at<double>(1, 2) = cy;
  intrinsics.at<double>(2, 2) = 1;
  
  return intrinsics;
}

cv::Mat stereo_rectifier::rotation_vec2mat(const cv::Mat &rot_vec) const {
  cv::Mat rot_mat;
  cv::Rodrigues(rot_vec, rot_mat);
  return rot_mat;
}

cv::Mat stereo_rectifier::parse_vector_as_mat(const std::vector<double>& vec) {
    cv::Mat mat(1, vec.size(), CV_64FC1);
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(double));
    return mat;
}

