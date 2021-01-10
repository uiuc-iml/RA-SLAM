#include <iostream>
#include <cstdint>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "top_level.h"

int main(int argc, char *argv[]) {
    std::cout << "Entering example main" << std::endl;

    /* Start Params Definition */
    camera_intrinsics_t l515_info;
    camera_intrinsics_t zed_left_info;
    camera_intrinsics_t zed_right_info;
    se3_pose_t l515_to_zed_transform;
    stereo_rect_extrinsics_t zed_ext_info;

    // dummy intrinsics for L515 camera
    l515_info.fx = 456.8617;
    l515_info.fy = 456.77127;
    l515_info.cx = 322.1042;
    l515_info.cy = 187.79485;
    l515_info.k1 = 0;
    l515_info.k2 = 0;
    l515_info.k3 = 0;
    l515_info.p1 = 0;
    l515_info.p2 = 0;
    l515_info.depth_scale = 4000;

    // dummy intrinsics for left ZED camera
    zed_left_info.fx = 0;
    zed_left_info.fy = 0;
    zed_left_info.cx = 0;
    zed_left_info.cy = 0;
    zed_left_info.k1 = 0;
    zed_left_info.k2 = 0;
    zed_left_info.k3 = 0;
    zed_left_info.p1 = 0;
    zed_left_info.p2 = 0;
    zed_left_info.depth_scale = 0;

    // dummy intrinsics for right ZED camera
    zed_right_info.fx = 0;
    zed_right_info.fy = 0;
    zed_right_info.cx = 0;
    zed_right_info.cy = 0;
    zed_right_info.k1 = 0;
    zed_right_info.k2 = 0;
    zed_right_info.k3 = 0;
    zed_right_info.p1 = 0;
    zed_right_info.p2 = 0;
    zed_right_info.depth_scale = 0;

    // dummy extrinsics for ZED/L515 camera
    for (int i = 0; i < 9; ++i) {
        l515_to_zed_transform.R[i] = 0;
    }
    // create an identity matrix
    l515_to_zed_transform.R[0] = 1;
    l515_to_zed_transform.R[4] = 1;
    l515_to_zed_transform.R[8] = 1;
    for (int i = 0; i < 3; ++i) {
        l515_to_zed_transform.t[i] = 0;
    }
    // some random timestamp...
    l515_to_zed_transform.timestamp = 0;

    // Stereo Rectification Parameters
    for (int i = 0; i < 3; ++i) {
        zed_ext_info.rotation[i] = 0;
        zed_ext_info.translation[i] = 0;
    }

    // initialize interface
    disinfect_slam_top_level my_disinfect_slam(
        l515_info,
        zed_left_info,
        zed_right_info,
        l515_to_zed_transform,
        zed_ext_info
    );

    // try feeding image
    cv::Mat rgb_img = cv::imread("../rgb.png", cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread("../depth.png", cv::IMREAD_ANYDEPTH);
    
    rgb_image_t l515_rgb;
    depth_image_t l515_depth;
    rgb_image_t zed_left;
    rgb_image_t zed_right;
    l515_rgb.data = &rgb_img;
    l515_depth.data = &depth_img;
    zed_left.data = &rgb_img;
    zed_right.data = &rgb_img;

    my_disinfect_slam.feed_camera_images(l515_rgb, l515_depth, zed_left, zed_right);

    // try extracting pose
    se3_pose_t my_pos = my_disinfect_slam.get_estimated_pose();
    std::cout << "SE3 R: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << my_pos.R[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "SE3 t: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << my_pos.t[i] << " ";
    }
    std::cout << std::endl;
    // try extracting reconstruction
    semantic_recon_t my_recon = my_disinfect_slam.get_full_semantic_reconstruction();
    std::cout << "printing semantic reconstruction" << std::endl;
    for (size_t i = 0; i < my_recon.data.size(); ++i) {
        std::cout << my_recon.data[i] << " ";
    }
    std::cout << std::endl;
}