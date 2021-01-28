#include "disinfect_slam/disinfect_slam.h"

DISINFSystem::DISINFSystem(
    std::string camera_config_path,
    std::string vocab_path,
    std::string seg_model_path,
    bool rendering_flag
) {
    std::shared_ptr<openvslam::config> cfg = GetAndSetConfig(camera_config_path);
    SLAM_ = std::make_shared<SLAMSystem>(cfg, vocab_path);
    SEG_ = std::make_shared<inference_engine>(seg_model_path);
    TSDF_ = std::make_shared<TSDFSystem>(0.01, 0.06, 4,
        GetIntrinsicsFromFile(camera_config_path), GetExtrinsicsFromFile(camera_config_path));

    depthmap_factor_ = GetDepthFactorFromFile(camera_config_path);

    SLAM_->startup();

    camera_pose_manager_ = std::make_shared<pose_manager>();

    if (rendering_flag) {
        RENDERER_ = std::make_shared<ImageRenderer>("tsdf", SLAM_, TSDF_, camera_config_path);
    }
}

DISINFSystem::~DISINFSystem() {
    SLAM_->shutdown();
}

void DISINFSystem::run() {
    RENDERER_->Run();
}

void DISINFSystem::feed_rgbd_frame(const cv::Mat & img_rgb, const cv::Mat & img_depth, int64_t timestamp) {
    cv::Mat my_img_rgb, my_img_depth; // local Mat that will be modified
    const SE3<float> posecam_P_world = camera_pose_manager_->query_pose(timestamp);
    cv::resize(img_rgb, my_img_rgb, cv::Size(), .5, .5);
    cv::resize(img_depth, my_img_depth, cv::Size(), .5, .5);
    my_img_depth.convertTo(my_img_depth, CV_32FC1, 1. / depthmap_factor_); // depth scale
    std::vector<cv::Mat> prob_map = SEG_->infer_one(my_img_rgb, false);
    TSDF_->Integrate(posecam_P_world, my_img_rgb, my_img_depth, prob_map[0], prob_map[1]);
}

void DISINFSystem::feed_stereo_frame(const cv::Mat & img_left, const cv::Mat & img_right, int64_t timestamp) {
    const pose_valid_tuple m = SLAM_->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
    const SE3<float> posecam_P_world(
        m.first(0, 0), m.first(0, 1), m.first(0, 2), m.first(0, 3),
        m.first(1, 0), m.first(1, 1), m.first(1, 2), m.first(1, 3),
        m.first(2, 0), m.first(2, 1), m.first(2, 2), m.first(2, 3),
        m.first(3, 0), m.first(3, 1), m.first(3, 2), m.first(3, 3)
    );
    if (m.second)
        camera_pose_manager_->register_valid_pose(timestamp, posecam_P_world);
}

SE3<float> DISINFSystem::query_camera_pose(const int64_t timestamp) {
    return this->camera_pose_manager_->query_pose(timestamp);
}

std::vector<VoxelSpatialTSDF> DISINFSystem::query_tsdf(const BoundingCube<float> &volumn) {
  return TSDF_->Query(volumn);
}
