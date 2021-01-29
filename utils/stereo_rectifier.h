#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>

struct CalibMono {
  double fx;
  double fy;
  double cx;
  double cy;
  std::vector<double> distortion;

  CalibMono(double fx, double fy, double cx, double cy, const std::vector<double>& distortion)
      : fx(fx), fy(fy), cx(cx), cy(cy), distortion(distortion) {}
};

struct CalibStereo {
  CalibMono left;
  CalibMono right;
  cv::Mat right_R_left;
  cv::Mat right_t_left;

  CalibStereo(const CalibMono& left, const CalibMono& right, const cv::Mat& right_R_left,
              const cv::Mat& right_t_left)
      : left(left), right(right), right_R_left(right_R_left), right_t_left(right_t_left) {}
};

class StereoRectifier {
 public:
  //! Constructor
  StereoRectifier(const cv::Size& img_size, const CalibStereo& calibration);
  explicit StereoRectifier(const YAML::Node& yaml_node);

  //! Destructor
  virtual ~StereoRectifier();

  //! Apply stereo-rectification
  void rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r, cv::Mat& out_img_l,
               cv::Mat& out_img_r) const;

  cv::Mat RectifiedIntrinsics() const;

 private:
  //! Parse std::vector as cv::Mat
  static cv::Mat ParseVectorAsMat(const std::vector<double>& vec);

  cv::Mat GetIntrinsicsAsMat(const CalibMono& calibration) const;

  cv::Mat RotationVecToMat(const cv::Mat& rot_vec) const;

  // matrix P2 in cv::stereoRectify which contains all camera parameters after
  // rectification
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
