#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>

/**
 * @brief template class for storing linear camera calibration
 *
 * @tparam T  paramter data type
 */
template <typename T>
struct CameraIntrinsics {
  const T fx;
  const T fy;
  const T cx;
  const T cy;

  /**
   * @brief construct linear calibration from parameter
   *
   * @param fx  x dimension focal length, in [pixel]
   * @param fy  y dimension focal length, in [pixel]
   * @param cx  x dimension principle point, in [pixel]
   * @param cy  y dimension principle point, in [pixel]
   */
  __device__ __host__ CameraIntrinsics(const T& fx, const T& fy, const T& cx, const T& cy)
      : fx(fx), fy(fy), cx(cx), cy(cy) {}

  /**
   * @brief optimized calibration matrix inverse
   *
   * @return inversed linear calibration matrix
   */
  __device__ __host__ CameraIntrinsics<T> Inverse() const {
    const T fx_inv = 1 / fx;
    const T fy_inv = 1 / fy;
    return CameraIntrinsics<T>(fx_inv, fy_inv, -cx * fx_inv, -cy * fy_inv);
  }

  /**
   * @brief project point on to pixel plane
   *
   * @param vec3  3D point to be projected
   *
   * @return homogeneous image plane coordinate
   */
  __device__ __host__ Eigen::Matrix<T, 3, 1> operator*(const Eigen::Matrix<T, 3, 1>& vec3) const {
    return Eigen::Matrix<T, 3, 1>(fx * vec3[0] + cx * vec3[2], fy * vec3[1] + cy * vec3[2],
                                  vec3[2]);
  }
};

class CameraParams {
 public:
  CameraIntrinsics<float> intrinsics;
  CameraIntrinsics<float> intrinsics_inv;
  int img_h;
  int img_w;

 public:
  __device__ __host__ CameraParams(const CameraIntrinsics<float>& intrinsics_, int img_h_,
                                   int img_w_)
      : img_h(img_h_),
        img_w(img_w_),
        intrinsics(intrinsics_),
        intrinsics_inv(intrinsics_.Inverse()) {}
};
