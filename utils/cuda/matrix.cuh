#pragma once

#include "vector.cuh"

template<typename T>
class Matrix3 {
 public:
  T m00, m01, m02;
  T m10, m11, m12;
  T m20, m21, m22;
 public:
  __device__ __host__ Matrix3<T>() {}

  __device__ __host__ explicit Matrix3<T>(const Matrix3<T> &others) 
      : m00(others.m00), m01(others.m01), m02(others.m02),
        m10(others.m10), m11(others.m11), m12(others.m12),
        m20(others.m20), m21(others.m21), m22(others.m22) {}

  __device__ __host__ static inline Matrix3<T> Zeros() { 
    return Matrix3<T>({ 0, 0, 0, 0, 0, 0, 0, 0, 0 }); 
  }

  __device__ __host__ static inline Matrix3<T> Ones() { 
    return Matrix3<T>({ 1, 1, 1, 1, 1, 1, 1, 1, 1 }); 
  }

  __device__ __host__ static inline Matrix3<T> Eyes() { 
    return Matrix3<T>({ 1, 0, 0, 0, 1, 0, 0, 0, 1 }); 
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

  __device__ __host__ inline Matrix3<T> operator+(const Matrix3<T> &rhs) const {
    Matrix3<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix3<T> operator+(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret += rhs;
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

  __device__ __host__ inline Matrix3<T> operator-(const Matrix3<T> &rhs) const {
    Matrix3<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator-(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix3<T> operator*=(const T &rhs) {
    m00 *= rhs; m01 *= rhs; m02 *= rhs;
    m10 *= rhs; m11 *= rhs; m12 *= rhs;
    m20 *= rhs; m21 *= rhs; m22 *= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix3<T> operator*(const T &rhs) const {
    Matrix3<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Vector3<T> operator*(const Vector3<T> &rhs) const {
    return Vector3<T>({
      m00 * rhs.x + m01 * rhs.y + m02 * rhs.z,
      m10 * rhs.x + m11 * rhs.y + m12 * rhs.z,
      m20 * rhs.x + m21 * rhs.y + m22 * rhs.z,
    });
  }
};

template<typename T>
__device__ __host__ inline Matrix3<T> operator+(const T &lhs, const Matrix3<T> &rhs) {
  Matrix3<T> ret(rhs);
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Matrix3<T> operator-(const T &lhs, const Matrix3<T> &rhs) {
  Matrix3<T> ret(rhs);
  return ret -= lhs;
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

  __device__ __host__ explicit Matrix4<T>(const Matrix4<T> &others) 
      : m00(others.m00), m01(others.m01), m02(others.m02), m03(others.m03),
        m10(others.m10), m11(others.m11), m12(others.m12), m13(others.m13),
        m20(others.m20), m21(others.m21), m22(others.m22), m23(others.m23), 
        m30(others.m30), m31(others.m31), m32(others.m32), m33(others.m33) {}

  __device__ __host__ static inline Matrix4<T> Zeros() { 
    return Matrix4<T>({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); 
  }

  __device__ __host__ static inline Matrix4<T> Ones() { 
    return Matrix4<T>({ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }); 
  }

  __device__ __host__ static inline Matrix4<T> Eyes() { 
    return Matrix4<T>({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }); 
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

  __device__ __host__ inline Matrix4<T> operator+(const Matrix4<T> &rhs) const {
    Matrix4<T> ret(*this);
    return ret += rhs;
  }

  __device__ __host__ inline Matrix4<T> operator+(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret += rhs;
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

  __device__ __host__ inline Matrix4<T> operator-(const Matrix4<T> &rhs) const {
    Matrix4<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator-(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret -= rhs;
  }

  __device__ __host__ inline Matrix4<T> operator*=(const T &rhs) {
    m00 *= rhs; m01 *= rhs; m02 *= rhs; m03 *= rhs;
    m10 *= rhs; m11 *= rhs; m12 *= rhs; m13 *= rhs;
    m20 *= rhs; m21 *= rhs; m22 *= rhs; m23 *= rhs;
    m30 *= rhs; m31 *= rhs; m32 *= rhs; m33 *= rhs;
    return *this;
  }

  __device__ __host__ inline Matrix4<T> operator*(const T &rhs) const {
    Matrix4<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Vector4<T> operator*(const Vector4<T> &rhs) const {
    return Vector4<T>({
      m00 * rhs.x + m01 * rhs.y + m02 * rhs.z + m03 * rhs.w,
      m10 * rhs.x + m11 * rhs.y + m12 * rhs.z + m13 * rhs.w,
      m20 * rhs.x + m21 * rhs.y + m22 * rhs.z + m23 * rhs.w,
      m30 * rhs.x + m31 * rhs.y + m32 * rhs.z + m33 * rhs.w,
    });
  }
};

template<typename T>
__device__ __host__ inline Matrix4<T> operator+(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(rhs);
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Matrix4<T> operator-(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(rhs);
  return ret -= lhs;
}

template<typename T>
__device__ __host__ inline Matrix4<T> operator*(const T &lhs, const Matrix4<T> &rhs) {
  Matrix4<T> ret(rhs);
  return ret *= lhs;
}

