#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/core/core.hpp>

// #include "utils/cuda/vector.cuh"

/**
 * struct containing rgb image and relevant information.
 * 
 * @var data        pointer to an uint8_t array of shape (H, W, C)
 *                  where C = 3 indicating R/G/B channel.
 * 
 * @var H           an unsigned integer indicating the height of the image.
 * 
 * @var W           an unsigned integer indicating the width of the image.
 * 
 * @var timestamp   a floating number containing UNIX timestamp of the image
 *                  indicating when it was captured by the perception module.
 */
typedef struct {
    cv::Mat* data;
    float timestamp;
} rgb_image_t;

/**
 * struct containing depth image and relevant information.
 * 
 * @var data        pointer to an uint16_t array of shape (H, W)
 *                  representing raw depth image captued from the camera.
 * 
 * @var H           an unsigned integer indicating the height of the image.
 * 
 * @var W           an unsigned integer indicating the width of the image.
 * 
 * @var timestamp   a floating number containing UNIX timestamp of the image
 *                  indicating when it was captured by the perception module.
 */
typedef struct {
    cv::Mat* data;
    float timestamp;
} depth_image_t;

/**
 * struct containing estimated SE3 pose estimated from ZED camera using
 * ORB-SLAM and relevant information
 * 
 * @var R           3x3 rotation matrix of row-major order.
 * 
 * @var t           3x1 translation vector
 * 
 *                       [[R[0], R[1], R[2]]
 *                  R =  [R[3], R[4], R[5]]     t = (t[0], t[1], t[2])^{T}
 *                       [R[6], R[7], R[8]]]
 * 
 * @var timestamp   a floating number containing a UNIX timestamp indicating the capture
 *                  time of the corresponding ZED images that were used to estimate
 *                  this pose.
 */
typedef struct {
    float R[9];
    float t[3];
    float timestamp;
} se3_pose_t;

/**
 * struct containing semantic reconstruction and relevant information
 * 
 * @var data        a pointer to an float array of shape (N, 5) where
 *                      N - number of voxels
 *                      5 - (x, y, z, tsdf, high_touch_prob)
 * 
 * @var size        equals to $N$ in the data pointer.
 * 
 * @var timestamp   a floating number containing a UNIX timestamp indicating the capture
 *                  time of the LAST L515 images that was used to update the reconstruction.
 */
typedef struct {
    // TODO: right now for simplicity, this is not pointer.
    // this needs to be changed in the future.
    std::vector<float> data;
    float timestamp;
} semantic_recon_t;

/**
 * struct containing camera intrinsics and distortion factors.
 * 
 * @var fx/fy/cx/cy Camera intrinsics.
 * 
 * @var k_i/p_i     Camera radial and tangential factors
 *                  as outlined at https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
 * 
 * @var depth_scale Depth factor to convert raw uint16_t camera depth map to float.
 *                      e.g. for L515, it should be 4000.
 */
typedef struct {
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float k3;
    float p1;
    float p2;
    float depth_scale;
} camera_intrinsics_t;

/**
 * struct containing ZED stereo camera extrinsics for rectification
 * 
 * @var rotation    Rotation represented in Euler
 * 
 * @var translation x/y/z translation.
 * 
 */
typedef struct {
    float rotation[3];
    float translation[3];
} stereo_rect_extrinsics_t;

class disinfect_slam_top_level {
    public:
        /**
         * 
         * Constrcutor. Start everything.
         * 
         * @param   l515_info   Intrinsics and Distortion Calibration for L515 camera.
         *                      See comments above camera_intrinsics_t
         * @param   zed_info    Intrinsics and Distortion Calibration for ZED camera.
         *                      See comments above camera_intrinsics_t
         * @param   l515_2_zed  SE3 transformation from L515 camera to ZED camera
         */
        disinfect_slam_top_level(
            camera_intrinsics_t         l515_info,
            camera_intrinsics_t         zed_left_info,
            camera_intrinsics_t         zed_right_info,
            se3_pose_t                  l515_2_zed,
            stereo_rect_extrinsics_t    zed_ext_info
        );

        /**
         * feed the latest (synchrnoized) images from cameras to perception module
         * 
         * @param   l515_rgb    Raw RGB image from the L515 camera
         * @param   l515_depth  Raw depth image from the L515 camera
         * @param   zed_left    Raw left RGB image from the ZED camera
         * @param   zed_right   Raw right RGB image from the ZED camera
         * 
         */
        void feed_camera_images(
            rgb_image_t l515_rgb,
            depth_image_t l515_depth,
            rgb_image_t zed_left,
            rgb_image_t zed_right
        );

        /**
         * get latest estimated pose (solely based on ZED camera input) from vSLAM
         * 
         * @return          see comments above se3_pose_t struct definition.
         *
         */
        se3_pose_t get_estimated_pose();

        /**
         * dump the entire semantically-fused reconstruction map
         * 
         * @return          see comments above semantic_recon_t struct definition.
         * 
         */
        semantic_recon_t get_full_semantic_reconstruction();

        /**
         * dump only PART of semantically-fused reconstruction map.
         * The part is given by user as a 3D bounding box.
         * 
         * @param bbox_vertex   the xyz coordinate of the lower-left-front vertex
         * @param bbox_size     the 3D dimension of the bounding box
         *                          - (length, width, height)
         * 
         * @return          see comments above semantic_recon_t struct definition.                 
         * 
         */
        // semantic_recon_t get_part_semantic_reconstruction(
        //     Vector3<float> bounding_box,
        //     Vector3<float> bbox_size
        // );
    private:
        camera_intrinsics_t l515_info;
        camera_intrinsics_t zed_left_info;
        camera_intrinsics_t zed_right_info;
        se3_pose_t l515_2_zed_transform;
        stereo_rect_extrinsics_t zed_ext_info;
};
