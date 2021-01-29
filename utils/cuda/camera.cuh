#pragma once

#include <cuda_runtime.h>

#include "utils/cuda/matrix.cuh"

/**
 * @brief template class for storing linear camera calibration
 *
 * @tparam T  paramter data type
 */
template <typename T>
class CameraIntrinsics : public Matrix3<T> {
 public:
  /**
   * @brief construct linear calibration from parameter
   *
   * @param fx  x dimension focal length, in [pixel]
   * @param fy  y dimension focal length, in [pixel]
   * @param cx  x dimension principle point, in [pixel]
   * @param cy  y dimension principle point, in [pixel]
   */
  __device__ __host__ CameraIntrinsics(const T& fx, const T& fy, const T& cx, const T& cy)
      : Matrix3<T>(fx, 0, cx, 0, fy, cy, 0, 0, 1) {}

  /**
   * @brief optimized calibration matrix inverse
   *
   * @return inversed linear calibration matrix
   */
  __device__ __host__ CameraIntrinsics<T> Inverse() const {
    const T fx_inv = 1 / this->m00;
    const T fy_inv = 1 / this->m11;
    const T cx = this->m02;
    const T cy = this->m12;
    return CameraIntrinsics<T>(fx_inv, fy_inv, -cx * fx_inv, -cy * fy_inv);
  }

  /**
   * @brief project point on to pixel plane
   *
   * @param vec3  3D point to be projected
   *
   * @return homogeneous image plane coordinate
   */
  __device__ __host__ Vector3<T> operator*(const Vector3<T>& vec3) const {
    return Vector3<T>(this->m00 * vec3.x + this->m02 * vec3.z,
                      this->m11 * vec3.y + this->m12 * vec3.z, vec3.z);
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
