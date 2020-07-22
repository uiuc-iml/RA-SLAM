#pragma once

#include <cuda_runtime.h>

#include "utils/cuda/vector.cuh"

template<typename T>
class Matrix3 {
 public:
  T m00, m01, m02;
  T m10, m11, m12;
  T m20, m21, m22;
 public:
  __device__ __host__ Matrix3<T>() {}

  __device__ __host__ explicit Matrix3<T>(const T &scalar)
    : m00(scalar), m01(scalar), m02(scalar),
      m10(scalar), m11(scalar), m12(scalar),
      m20(scalar), m21(scalar), m22(scalar) {}

  __device__ __host__ Matrix3<T>(const T &a00, const T &a01, const T &a02,
                                 const T &a10, const T &a11, const T &a12,
                                 const T &a20, const T &a21, const T &a22)
    : m00(a00), m01(a01), m02(a02),
      m10(a10), m11(a11), m12(a12),
      m20(a20), m21(a21), m22(a22) {}

  __device__ __host__ static inline Matrix3<T> Identity() { 
    Matrix3<T> mat(0);
    mat.m00 = mat.m11 = mat.m22 = 1;
    return mat; 
  }

  template<typename Tout>
  __device__ __host__ inline Matrix3<Tout> cast() const {
    return Matrix3<Tout>(
      static_cast<Tout>(m00), static_cast<Tout>(m01), static_cast<Tout>(m02),
      static_cast<Tout>(m10), static_cast<Tout>(m11), static_cast<Tout>(m12),
      static_cast<Tout>(m20), static_cast<Tout>(m21), static_cast<Tout>(m22)
    );
  }

  __device__ __host__ inline Matrix3<T>& operator+=(const Matrix3<T> &rhs) {
    m00 += rhs.m00; m01 += rhs.m01; m02 += rhs.m02;
    m10 += rhs.m10; m11 += rhs.m11; m12 += rhs.m12;
    m20 += rhs.m20; m21 += rhs.m21; m22 += rhs.m22;
    return *this;
  }

  __device__ __host__ inline Matrix3<T>& operator+=(const T &rhs) {
    m00 += rhs; m01 += rhs; m02 += rhs;
    m10 += rhs; m11 += rhs; m12 += rhs;
    m20 += rhs; m21 += rhs; m22 += rhs;
    return *this;
  }

  __device__ __host__ inline Matrix3<T>& operator-=(const Matrix3<T> &rhs) {
    m00 -= rhs.m00; m01 -= rhs.m01; m02 -= rhs.m02;
    m10 -= rhs.m10; m11 -= rhs.m11; m12 -= rhs.m12;
    m20 -= rhs.m20; m21 -= rhs.m21; m22 -= rhs.m22;
    return *this;
  }

  __device__ __host__ inline Matrix3<T>& operator-=(const T &rhs) {
    m00 -= rhs; m01 -= rhs; m02 -= rhs;
    m10 -= rhs; m11 -= rhs; m12 -= rhs;
    m20 -= rhs; m21 -= rhs; m22 -= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix3<T>& operator*=(const Matrix3<T> &rhs) {
    const Matrix3<T> old(*this);
    m00 = old.m00 * rhs.m00 + old.m01 * rhs.m10 + old.m02 * rhs.m20;
    m01 = old.m00 * rhs.m01 + old.m01 * rhs.m11 + old.m02 * rhs.m21;
    m02 = old.m00 * rhs.m02 + old.m01 * rhs.m12 + old.m02 * rhs.m22;
    m10 = old.m10 * rhs.m00 + old.m11 * rhs.m10 + old.m12 * rhs.m20;
    m11 = old.m10 * rhs.m01 + old.m11 * rhs.m11 + old.m12 * rhs.m21;
    m12 = old.m10 * rhs.m02 + old.m11 * rhs.m12 + old.m12 * rhs.m22;
    m20 = old.m20 * rhs.m00 + old.m21 * rhs.m10 + old.m22 * rhs.m20;
    m21 = old.m20 * rhs.m01 + old.m21 * rhs.m11 + old.m22 * rhs.m21;
    m22 = old.m20 * rhs.m02 + old.m21 * rhs.m12 + old.m22 * rhs.m22;
    return *this;
  }

  __device__ __host__ inline Matrix3<T> operator*=(const T &rhs) {
    m00 *= rhs; m01 *= rhs; m02 *= rhs;
    m10 *= rhs; m11 *= rhs; m12 *= rhs;
    m20 *= rhs; m21 *= rhs; m22 *= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix3<T> operator/=(const T &rhs) {
    m00 /= rhs; m01 /= rhs; m02 /= rhs;
    m10 /= rhs; m11 /= rhs; m12 /= rhs;
    m20 /= rhs; m21 /= rhs; m22 /= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix3<T> operator+(const Matrix3<T> &rhs) const {
    Matrix3<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix3<T> operator+(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix3<T> operator-(const Matrix3<T> &rhs) const {
    Matrix3<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator-(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator-() const {
    return (T)0 - *this;
  }

  __device__ __host__ inline Matrix3<T> operator*(const Matrix3<T> &rhs) const {
    Matrix3<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator*(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator/(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret /= rhs;
  }

  __device__ __host__ inline bool operator==(const Matrix3<T> &rhs) const {
    return m00 == rhs.m00 && m01 == rhs.m01 && m02 == rhs.m02 &&
           m10 == rhs.m10 && m11 == rhs.m11 && m12 == rhs.m12 &&
           m20 == rhs.m20 && m21 == rhs.m21 && m22 == rhs.m22;
  }

  __device__ __host__ inline bool operator!=(const Matrix3<T> &rhs) const {
    return !operator==(rhs);
  }

  __device__ __host__ inline Vector3<T> operator*(const Vector3<T> &rhs) const {
    return Vector3<T>(
      m00 * rhs.x + m01 * rhs.y + m02 * rhs.z,
      m10 * rhs.x + m11 * rhs.y + m12 * rhs.z,
      m20 * rhs.x + m21 * rhs.y + m22 * rhs.z
    );
  }
};

template<typename T>
__device__ __host__ inline Matrix3<T> operator+(const T &lhs, const Matrix3<T> &rhs) {
  Matrix3<T> ret(rhs);
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Matrix3<T> operator-(const T &lhs, const Matrix3<T> &rhs) {
  Matrix3<T> ret(lhs);
  return ret -= rhs;
}

template<typename T>
__device__ __host__ inline Matrix3<T> operator*(const T &lhs, const Matrix3<T> &rhs) {
  Matrix3<T> ret(rhs);
  return ret *= lhs;
}

template<typename T>
class Matrix4 {
 public:
  T m00, m01, m02, m03;
  T m10, m11, m12, m13;
  T m20, m21, m22, m23;
  T m30, m31, m32, m33;
 public:
  __device__ __host__ Matrix4<T>() {}

  __device__ __host__ explicit Matrix4<T>(const T &scalar)
    : m00(scalar), m01(scalar), m02(scalar), m03(scalar),
      m10(scalar), m11(scalar), m12(scalar), m13(scalar),
      m20(scalar), m21(scalar), m22(scalar), m23(scalar),
      m30(scalar), m31(scalar), m32(scalar), m33(scalar) {}

  __device__ __host__ Matrix4<T>(const T &a00, const T &a01, const T &a02, const T &a03,
                                 const T &a10, const T &a11, const T &a12, const T &a13,
                                 const T &a20, const T &a21, const T &a22, const T &a23,
                                 const T &a30, const T &a31, const T &a32, const T &a33)
    : m00(a00), m01(a01), m02(a02), m03(a03),
      m10(a10), m11(a11), m12(a12), m13(a13),
      m20(a20), m21(a21), m22(a22), m23(a23),
      m30(a30), m31(a31), m32(a32), m33(a33) {}

  __device__ __host__ static inline Matrix4<T> Identity() { 
    Matrix4<T> mat(0);
    mat.m00 = mat.m11 = mat.m22 = mat.m33 = 1;
    return mat; 
  }

  template<typename Tout>
  __device__ __host__ inline Matrix4<Tout> cast() const {
    return Matrix4<Tout>(
      static_cast<Tout>(m00), static_cast<Tout>(m01), static_cast<Tout>(m02), static_cast<Tout>(m03),
      static_cast<Tout>(m10), static_cast<Tout>(m11), static_cast<Tout>(m12), static_cast<Tout>(m13),
      static_cast<Tout>(m20), static_cast<Tout>(m21), static_cast<Tout>(m22), static_cast<Tout>(m23),
      static_cast<Tout>(m30), static_cast<Tout>(m31), static_cast<Tout>(m32), static_cast<Tout>(m33)
    );
  }

  __device__ __host__ inline Matrix4<T>& operator+=(const Matrix4<T> &rhs) {
    m00 += rhs.m00; m01 += rhs.m01; m02 += rhs.m02; m03 += rhs.m03;
    m10 += rhs.m10; m11 += rhs.m11; m12 += rhs.m12; m13 += rhs.m13;
    m20 += rhs.m20; m21 += rhs.m21; m22 += rhs.m22; m23 += rhs.m23;
    m30 += rhs.m30; m31 += rhs.m31; m32 += rhs.m32; m33 += rhs.m33;
    return *this;
  }

  __device__ __host__ inline Matrix4<T>& operator+=(const T &rhs) {
    m00 += rhs; m01 += rhs; m02 += rhs; m03 += rhs;
    m10 += rhs; m11 += rhs; m12 += rhs; m13 += rhs;
    m20 += rhs; m21 += rhs; m22 += rhs; m23 += rhs;
    m30 += rhs; m31 += rhs; m32 += rhs; m33 += rhs;
    return *this;
  }

  __device__ __host__ inline Matrix4<T>& operator-=(const Matrix4<T> &rhs) {
    m00 -= rhs.m00; m01 -= rhs.m01; m02 -= rhs.m02; m03 -= rhs.m03;
    m10 -= rhs.m10; m11 -= rhs.m11; m12 -= rhs.m12; m13 -= rhs.m13;
    m20 -= rhs.m20; m21 -= rhs.m21; m22 -= rhs.m22; m23 -= rhs.m23;
    m30 -= rhs.m30; m31 -= rhs.m31; m32 -= rhs.m32; m33 -= rhs.m33;
    return *this;
  }

  __device__ __host__ inline Matrix4<T>& operator-=(const T &rhs) {
    m00 -= rhs; m01 -= rhs; m02 -= rhs; m03 -= rhs;
    m10 -= rhs; m11 -= rhs; m12 -= rhs; m13 -= rhs;
    m20 -= rhs; m21 -= rhs; m22 -= rhs; m23 -= rhs;
    m30 -= rhs; m31 -= rhs; m32 -= rhs; m33 -= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix4<T>& operator*=(const Matrix4<T> &rhs) {
    const Matrix4<T> old(*this);
    m00 = old.m00 * rhs.m00 + old.m01 * rhs.m10 + old.m02 * rhs.m20 + old.m03 * rhs.m30;
    m01 = old.m00 * rhs.m01 + old.m01 * rhs.m11 + old.m02 * rhs.m21 + old.m03 * rhs.m31;
    m02 = old.m00 * rhs.m02 + old.m01 * rhs.m12 + old.m02 * rhs.m22 + old.m03 * rhs.m32;
    m03 = old.m00 * rhs.m03 + old.m01 * rhs.m13 + old.m02 * rhs.m23 + old.m03 * rhs.m33;
    m10 = old.m10 * rhs.m00 + old.m11 * rhs.m10 + old.m12 * rhs.m20 + old.m13 * rhs.m30;
    m11 = old.m10 * rhs.m01 + old.m11 * rhs.m11 + old.m12 * rhs.m21 + old.m13 * rhs.m31;
    m12 = old.m10 * rhs.m02 + old.m11 * rhs.m12 + old.m12 * rhs.m22 + old.m13 * rhs.m32;
    m13 = old.m10 * rhs.m03 + old.m11 * rhs.m13 + old.m12 * rhs.m23 + old.m13 * rhs.m33;
    m20 = old.m20 * rhs.m00 + old.m21 * rhs.m10 + old.m22 * rhs.m20 + old.m23 * rhs.m30;
    m21 = old.m20 * rhs.m01 + old.m21 * rhs.m11 + old.m22 * rhs.m21 + old.m23 * rhs.m31;
    m22 = old.m20 * rhs.m02 + old.m21 * rhs.m12 + old.m22 * rhs.m22 + old.m23 * rhs.m32;
    m23 = old.m20 * rhs.m03 + old.m21 * rhs.m13 + old.m22 * rhs.m23 + old.m23 * rhs.m33;
    m30 = old.m30 * rhs.m00 + old.m31 * rhs.m10 + old.m32 * rhs.m20 + old.m33 * rhs.m30;
    m31 = old.m30 * rhs.m01 + old.m31 * rhs.m11 + old.m32 * rhs.m21 + old.m33 * rhs.m31;
    m32 = old.m30 * rhs.m02 + old.m31 * rhs.m12 + old.m32 * rhs.m22 + old.m33 * rhs.m32;
    m33 = old.m30 * rhs.m03 + old.m31 * rhs.m13 + old.m32 * rhs.m23 + old.m33 * rhs.m33;
    return *this;
  }

  __device__ __host__ inline Matrix4<T> operator*=(const T &rhs) {
    m00 *= rhs; m01 *= rhs; m02 *= rhs; m03 *= rhs;
    m10 *= rhs; m11 *= rhs; m12 *= rhs; m13 *= rhs;
    m20 *= rhs; m21 *= rhs; m22 *= rhs; m23 *= rhs;
    m30 *= rhs; m31 *= rhs; m32 *= rhs; m33 *= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix4<T> operator/=(const T &rhs) {
    m00 /= rhs; m01 /= rhs; m02 /= rhs; m03 /= rhs;
    m10 /= rhs; m11 /= rhs; m12 /= rhs; m13 /= rhs;
    m20 /= rhs; m21 /= rhs; m22 /= rhs; m23 /= rhs;
    m30 /= rhs; m31 /= rhs; m32 /= rhs; m33 /= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix4<T> operator+(const Matrix4<T> &rhs) const {
    Matrix4<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix4<T> operator+(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix4<T> operator-(const Matrix4<T> &rhs) const {
    Matrix4<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator-(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator-() const {
    return (T)0 - *this;
  }

  __device__ __host__ inline Matrix4<T> operator*(const Matrix4<T> &rhs) const {
    Matrix4<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator*(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator/(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret /= rhs;
  }

  __device__ __host__ inline Vector4<T> operator*(const Vector4<T> &rhs) const {
    return Vector4<T>(
      m00 * rhs.x + m01 * rhs.y + m02 * rhs.z + m03 * rhs.w,
      m10 * rhs.x + m11 * rhs.y + m12 * rhs.z + m13 * rhs.w,
      m20 * rhs.x + m21 * rhs.y + m22 * rhs.z + m23 * rhs.w,
      m30 * rhs.x + m31 * rhs.y + m32 * rhs.z + m33 * rhs.w
    );
  }

  __device__ __host__ inline bool operator==(const Matrix4<T> &rhs) const {
    return m00 == rhs.m00 && m01 == rhs.m01 && m02 == rhs.m02 && m03 == rhs.m03 &&
           m10 == rhs.m10 && m11 == rhs.m11 && m12 == rhs.m12 && m13 == rhs.m13 &&
           m20 == rhs.m20 && m21 == rhs.m21 && m22 == rhs.m22 && m23 == rhs.m23 &&
           m30 == rhs.m30 && m31 == rhs.m31 && m32 == rhs.m32 && m33 == rhs.m33;
  }

  __device__ __host__ inline bool operator!=(const Matrix4<T> &rhs) const {
    return !operator==(rhs);
  }
};

template<typename T>
__device__ __host__ inline Matrix4<T> operator+(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(rhs);
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Matrix4<T> operator-(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(lhs);
  return ret -= rhs;
}

template<typename T>
__device__ __host__ inline Matrix4<T> operator*(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(rhs);
  return ret *= lhs;
}

