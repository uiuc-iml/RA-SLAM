#pragma once

#include <cuda_runtime.h>

#include "utils/cuda/matrix.cuh"

template<typename T>
class SO3 : public Matrix3<T> {
 public:
  using Matrix3<T>::Matrix3;

  __device__ __host__ SO3(const Matrix3<T> &mat3) : Matrix3<T>(mat3) {}

  __device__ __host__ inline SO3<T> Inverse() const {
    return SO3<T>(
      this->m00, this->m10, this->m20,
      this->m01, this->m11, this->m21,
      this->m02, this->m12, this->m22
    );
  }
};

template<typename T>
class SE3 : public Matrix4<T> {
 public:
  using Matrix4<T>::Matrix4;

  __device__ __host__ SE3(const Matrix4<T> &mat4) : Matrix4<T>(mat4) {}

  __device__ __host__ SE3<T>(const SO3<T> &rot, const Vector3<T> &trans)
    : Matrix4<T>(
      rot.m00, rot.m01, rot.m02, trans.x,
      rot.m10, rot.m11, rot.m12, trans.y,
      rot.m20, rot.m21, rot.m22, trans.z,
      0,       0,       0,       1
    ) {}

  __device__ __host__ inline SE3<T> Inverse() const {
    const SO3<T> r_inv(
      this->m00, this->m10, this->m20,
      this->m01, this->m11, this->m21,
      this->m02, this->m12, this->m22
    );
    const Vector3<T> t(this->m03, this->m13, this->m23);
    return SE3<T>(r_inv, -r_inv * t);
  }

  __device__ __host__ inline SO3<T> GetR() const {
    return SO3<T>(
      this->m00, this->m01, this->m02,
      this->m10, this->m11, this->m12,
      this->m20, this->m21, this->m22
    );
  }

  __device__ __host__ inline Vector3<T> GetT() const {
    return Vector3<T>(this->m03, this->m13, this->m23);
  }

  __device__ __host__ inline Vector3<T> Apply(const Vector3<T> &vec3) const {
    return Vector3<T>(
      this->m00 * vec3.x + this->m01 * vec3.y + this->m02 * vec3.z + this->m03,
      this->m10 * vec3.x + this->m11 * vec3.y + this->m12 * vec3.z + this->m13,
      this->m20 * vec3.x + this->m21 * vec3.y + this->m22 * vec3.z + this->m23
    );
  }

  __device__ __host__ SE3<T> operator*(const SE3<T> others) const {
    const SO3<T> R1 = GetR();
    const SO3<T> R2 = others.GetR();
    const Vector3<T> T1 = GetT();
    const Vector3<T> T2 = others.GetT();
    return SE3<T>(R1 * R2, R1 * T2 + T1);
  }

  __device__ __host__ Vector4<T> operator*(const Vector4<T> &others) const {
    return Vector4<T>(Apply(Vector3<T>(others)));
  }
};
