#include "utils/offline_data_provider/scannet_sens_reader.h"

#include <assert.h>

#include <iostream>
#include <memory>

scannet_sens_reader::scannet_sens_reader(const string& sens_filepath) : sd_(sens_filepath) {
  // Constructor code goes here
}

CameraIntrinsics<float> scannet_sens_reader::scannet_sens_reader::get_camera_intrinsics() {
  float fx = sd_.m_calibrationDepth.m_intrinsic.matrix2[0][0];
  float fy = sd_.m_calibrationDepth.m_intrinsic.matrix2[1][1];
  float cx = sd_.m_calibrationDepth.m_intrinsic.matrix2[0][2];
  float cy = sd_.m_calibrationDepth.m_intrinsic.matrix2[1][2];
  return CameraIntrinsics<float>(fx, fy, cx, cy);
}

SE3<float> scannet_sens_reader::get_camera_extrinsics() {
  /* For the ScanNet dataset, extrinsics must be identity matrix. */
  /* assert R is identity matrix and SE3[3, 3] is 1 */
  for (int i = 0; i < 4; ++i) {
    assert(sd_.m_calibrationDepth.m_extrinsic.matrix2[i][i] == 1);
  }

  /* assert that the rest of elements are zero */
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i != j) {
        assert(sd_.m_calibrationDepth.m_extrinsic.matrix2[i][j] == 0);
      }
    }
  }
  return SE3<float>::Identity();
}

float scannet_sens_reader::get_depth_map_factor() { return (float)(sd_.m_depthShift); }

void scannet_sens_reader::get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx) {
  unsigned short* depth_data = sd_.decompressDepthAlloc(frame_idx);

  /* Depth resolution are 640 * 480 in ScanNet*/
  unsigned int depth_width = sd_.m_depthWidth;
  unsigned int depth_height = sd_.m_depthHeight;
  assert((int)depth_height == get_height());
  assert((int)depth_width == get_width());
  *depth_img = cv::Mat(cv::Size(depth_width, depth_height), CV_16UC1, (void*)(depth_data),
                       cv::Mat::AUTO_STEP)
                   .clone();
  std::free(depth_data);
}

void scannet_sens_reader::get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx) {
  ml::vec3uc* color_data = sd_.decompressColorAlloc(frame_idx);

  /* color width and color height are actually 1296 * 968 in ScanNet */
  unsigned int color_width = sd_.m_colorWidth;
  unsigned int color_height = sd_.m_colorHeight;
  *rgb_img =
      cv::Mat(cv::Size(color_width, color_height), CV_8UC3, (void*)(color_data), cv::Mat::AUTO_STEP)
          .clone();

  /* for consistency with depth, we rescale it to 640 * 480 */
  cv::resize(*rgb_img, *rgb_img, cv::Size(get_width(), get_height()));
  std::free(color_data);
}

SE3<float> scannet_sens_reader::get_camera_pose_by_id(int frame_idx) {
  /* SE3 matrix has 16 elements */
  const float* pose_arr_pointer = sd_.m_frames[frame_idx].getCameraToWorld().matrix;
  std::vector<float> pose_arr(pose_arr_pointer, pose_arr_pointer + 16);
  Eigen::Matrix<float, 4, 4> col_major_temp =
      Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(pose_arr.data());
  return SE3<float>(col_major_temp).Inverse();
}

int scannet_sens_reader::get_size() { return (int)(sd_.m_frames.size()); }

int scannet_sens_reader::get_width() { return 640; }

int scannet_sens_reader::get_height() { return 480; }
