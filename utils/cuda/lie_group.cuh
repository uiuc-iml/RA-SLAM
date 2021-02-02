#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>

template <typename T>
class SE3 {
 public:
  __device__ __host__ SE3<T>(){};

  __device__ __host__ SE3<T>(const Eigen::Quaternion<T>& rot, const Eigen::Matrix<T, 3, 1>& trans)
      : R_(rot), t_(trans) {}

  __device__ __host__ explicit SE3<T>(const Eigen::Matrix<T, 4, 4>& mat)
      : R_(mat.template topLeftCorner<3, 3>()), t_(mat.template topRightCorner<3, 1>()) {}

  __device__ __host__ explicit SE3<T>(const Eigen::Matrix<T, 3, 4>& mat)
      : R_(mat.template topLeftCorner<3, 3>()), t_(mat.template topRightCorner<3, 1>()) {}

  __device__ __host__ static SE3<T> Identity() {
    return SE3<T>(Eigen::Quaternion<T>::Identity(), Eigen::Matrix<T, 3, 1>::Zero());
  }

  __device__ __host__ inline SE3<T> Inverse() const {
    return SE3<T>(R_.inverse(), R_.inverse() * (-t_));
  }

  __device__ __host__ inline Eigen::Quaternion<T> GetR() const { return R_; }

  __device__ __host__ inline Eigen::Matrix<T, 3, 1> GetT() const { return t_; }

  __device__ __host__ inline Eigen::Matrix<T, 3, 1> Apply(
      const Eigen::Matrix<T, 3, 1>& vec3) const {
    return R_ * vec3 + t_;
  }

  __device__ __host__ inline SE3<T> operator*(const SE3<T> others) const {
    return SE3<T>(R_ * others.R_, R_ * others.t_ + t_);
  }

 private:
  Eigen::Quaternion<T> R_;
  Eigen::Matrix<T, 3, 1> t_;
};
