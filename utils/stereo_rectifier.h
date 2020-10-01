#pragma once

#include <opencv2/imgproc.hpp>

#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

struct calib_mono {
  double fx;
  double fy;
  double cx;
  double cy;
  std::vector<double> distortion;

  calib_mono(double fx, double fy, double cx, double cy, const std::vector<double> &distortion)
    : fx(fx), fy(fy), cx(cx), cy(cy), distortion(distortion) {}
};

struct calib_stereo {
  calib_mono left;
  calib_mono right;
  cv::Mat right_R_left;
  cv::Mat right_T_left;

  calib_stereo(const calib_mono &left, const calib_mono &right,
               const cv::Mat &right_R_left, const cv::Mat &right_T_left)
    : left(left), right(right), right_R_left(right_R_left), right_T_left(right_T_left) {}
};

class stereo_rectifier {
 public:
  //! Constructor
  stereo_rectifier(const cv::Size &img_size, const calib_stereo &calibration);
  explicit stereo_rectifier(const YAML::Node &yaml_node);

  //! Destructor
  virtual ~stereo_rectifier();

  //! Apply stereo-rectification
  void rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r,
               cv::Mat& out_img_l, cv::Mat& out_img_r) const;

  const cv::Mat get_rectified_intrinsics() const;

 private:
  //! Parse std::vector as cv::Mat
  static cv::Mat parse_vector_as_mat(const std::vector<double>& vec);

  cv::Mat get_intrinsics(const calib_mono &calibration) const;

  cv::Mat rotation_vec2mat(const cv::Mat &rot_vec) const;

  // matrix P2 in cv::stereoRectify which contains all camera parameters after rectification
  cv::Mat cam_rect_matrix_;
  // matrix Q in cv::stereoRectify
  cv::Mat reproj_mat_;
  //! undistortion map for x-axis in left image
  cv::Mat undist_map_x_l_;
  //! undistortion map for y-axis in left image
  cv::Mat undist_map_y_l_;
  //! undistortion map for x-axis in right image
  cv::Mat undist_map_x_r_;
  //! undistortion map for y-axis in right image
  cv::Mat undist_map_y_r_;
};


