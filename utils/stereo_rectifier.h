#pragma once

#include <opencv2/imgproc.hpp>

#include <memory>
#include <yaml-cpp/yaml.h>

class stereo_rectifier {
public:
    //! Constructor
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

    cv::Mat get_intrinsics(double fx, double fy, double cx, double cy) const;

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


