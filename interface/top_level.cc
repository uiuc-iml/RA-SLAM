#include "top_level.h"

#include <iostream>

disinfect_slam_top_level::disinfect_slam_top_level(
    camera_intrinsics_t         l515_info,
    camera_intrinsics_t         zed_left_info,
    camera_intrinsics_t         zed_right_info,
    se3_pose_t                  l515_2_zed,
    stereo_rect_extrinsics_t    zed_ext_info
) {
    this->l515_info = l515_info;
    this->zed_left_info = zed_left_info;
    this->zed_right_info = zed_right_info;
    this->l515_2_zed_transform = l515_2_zed;
    this->zed_ext_info = zed_ext_info;

    std::cout << "Perception SLAM Module Initialized" << std::endl;
}

void disinfect_slam_top_level::feed_camera_images(
    rgb_image_t l515_rgb,
    depth_image_t l515_depth,
    rgb_image_t zed_left,
    rgb_image_t zed_right
) {
    std::cout << "Receiving images" << std::endl;
    std::cout << "L515 RGB Row: " << l515_rgb.data->rows << std::endl;
    std::cout << "L515 RGB Col: " << l515_rgb.data->cols << std::endl;
    std::cout << "L515 RGB Timestamp: " << l515_rgb.timestamp << std::endl;
    std::cout << "L515 Depth Row: " << l515_rgb.data->rows << std::endl;
    std::cout << "L515 Depth Col: " << l515_rgb.data->cols << std::endl;
    std::cout << "L515 Depth Dtype: " << l515_rgb.data->type() << std::endl;
    std::cout << "L515 Depth Timesteamp: " << l515_rgb.timestamp << std::endl;
    std::cout << "ZED LEFT RGB Row: " << l515_rgb.data->rows << std::endl;
    std::cout << "Zed LEFT RGB Col: " << l515_rgb.data->cols << std::endl;
    std::cout << "Zed LEFT RGB Timestamp: " << l515_rgb.timestamp << std::endl;
    std::cout << "ZED RIGHT RGB Row: " << l515_rgb.data->rows << std::endl;
    std::cout << "ZED RIGHT RGB Col: " << l515_rgb.data->cols << std::endl;
    std::cout << "ZED RIGHT RGB Timestamp: " << l515_rgb.timestamp << std::endl;
    std::cout << "End Receiving images" << std::endl;
}

se3_pose_t disinfect_slam_top_level::get_estimated_pose() {
    se3_pose_t ret;
    for (int i = 0; i < 9; ++i) {
        ret.R[i] = 0;
    }
    // create an identity matrix
    ret.R[0] = 1;
    ret.R[4] = 1;
    ret.R[8] = 1;
    for (int i = 0; i < 3; ++i) {
        ret.t[i] = 0;
    }
    // some random timestamp...
    ret.timestamp = 1600978062.1043277;
    return ret;
}

semantic_recon_t disinfect_slam_top_level::get_full_semantic_reconstruction() {
    semantic_recon_t ret;
    int size = 2;
    // 5 here refers to x/y/z/tsdf/ht_prob
    float toy_data[size][5] = {
        {0, 0, 0.1, 0.001, 1},
        {0.1, 0.2, 0.3, 0.002, 0.4}
    };
    for (uint32_t i = 0; i < size; ++i){
        for (int j = 0; j < 5; ++j) {
            ret.data.push_back(toy_data[i][j]);
        }
    }
    // some random timestamp...
    ret.timestamp = 1600978062.1043277;
    return ret;
}