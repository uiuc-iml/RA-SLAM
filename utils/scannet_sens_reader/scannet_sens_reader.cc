#include "utils/scannet_sens_reader/scannet_sens_reader.h"

#include <iostream>
#include <assert.h>

scannet_sens_reader::scannet_sens_reader(const string & sens_filepath)
    : sd_(sens_filepath) {
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
  // For the ScanNet dataset, extrinsics must be identity matrix.
  // assert R is identity matrix and SE3[3, 3] is 1
  for (int i = 0; i < 4; ++i) {
    assert (sd_.m_calibrationDepth.m_extrinsic.matrix2[i][i] == 1);
  }
  // assert that the rest of elements are zero
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i != j) {
        assert (sd_.m_calibrationDepth.m_extrinsic.matrix2[i][j] == 0);
      }
    }
  }
  return SE3<float>::Identity();
}

float scannet_sens_reader::get_depth_map_factor() {
  return (float)(sd_.m_depthShift);
}

void scannet_sens_reader::get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx) {
  unsigned short* depth_data = sd_.decompressDepthAlloc(frame_idx);
  unsigned int depth_width = sd_.m_depthWidth;
	unsigned int depth_height = sd_.m_depthHeight;
  *depth_img = cv::Mat(cv::Size(depth_width, depth_height), CV_16UC1,
                         (void*)(depth_data), cv::Mat::AUTO_STEP);
  // std::free(depth_data);
}

void scannet_sens_reader::get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx) {
  ml::vec3uc* color_data = sd_.decompressColorAlloc(frame_idx);
  std::cout << "Size of ml::vec3uc: " << sizeof(ml::vec3uc) << std::endl;
  unsigned int color_width = sd_.m_colorWidth;
	unsigned int color_height = sd_.m_colorHeight;
  // TODO: check memory leak?
  *rgb_img = cv::Mat(cv::Size(color_width, color_height), CV_8UC3,
                       (void*)(color_data), cv::Mat::AUTO_STEP);
  // std::free(color_data);
}

int scannet_sens_reader::get_size() {
  return (int)(sd_.m_frames.size());
}