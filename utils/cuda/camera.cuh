#pragma once

#include "utils/cuda/matrix.cuh"

template<typename T>
class CameraIntrinsics : public Matrix3<T> {
 public:
  __device__ __host__ CameraIntrinsics(const T &fx, const T &fy, const T &cx, const T &cy) 
    : Matrix3<T>(
      fx, 0,  cx,
      0,  fy, cy,
      0,  0,  1
    ) {}

  __device__ __host__ Matrix3<T> Inverse() const {
    const T fx_inv = 1 / this->m00;
    const T fy_inv = 1 / this->m11;
    const T cx = this->m02;
    const T cy = this->m12;
    return Matrix3<T>(
      fx_inv, 0,      -cx * fx_inv,
      0,      fy_inv, -cy * fy_inv,
      0,      0,      1  
    );
  }
};
