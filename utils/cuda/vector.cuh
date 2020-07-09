#pragma once

template<typename T>
class Vector3 {
 public:
  T x, y, z;
 public:
  __device__ __host__ Vector3<T>() {}

  __device__ __host__ Vector3<T>(const T &x_, const T &y_, const T &z_) 
      : x(x_), y(y_), z(z_) {}

  __device__ __host__ static inline Vector3<T> Zeros() { return Vector3<T>({ 0, 0, 0 }); }

  __device__ __host__ static inline Vector3<T> Ones() { return Vector3<T>({ 1, 1, 1 }); }

  __device__ __host__ inline Vector3<T>& operator+=(const Vector3<T> &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator+=(const T &rhs) {
    x += rhs; y += rhs; z += rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator-=(const Vector3<T> &rhs) {
    x -= rhs.x; y -= rhs.y; z -= rhs.z;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator-=(const T &rhs) {
    x -= rhs; y -= rhs; z -= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator*=(const T &rhs) {
    x *= rhs; y *= rhs; z *= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator/=(const T &rhs) {
    x /= rhs; y /= rhs; z /= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator&=(const T &rhs) {
    x &= rhs; y &= rhs; z &=rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator<<=(const T &rhs) {
    x <<= rhs; y <<= rhs; z <<= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T>& operator>>=(const T &rhs) {
    x >>= rhs; y >>= rhs; z >>= rhs;
    return *this;
  }

  __device__ __host__ inline Vector3<T> operator+(const Vector3<T> &rhs) const {
    Vector3<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector3<T> operator+(const T &rhs) const {
    Vector3<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector3<T> operator-(const Vector3<T> &rhs) const {
    Vector3<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector3<T> operator-(const T &rhs) const {
    Vector3<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector3<T> operator*(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline Vector3<T> operator/(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret /= rhs;
  }

  __device__ __host__ inline Vector3<T> operator&(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret &= rhs;
  }

  __device__ __host__ inline Vector3<T> operator<<(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret <<= rhs;
  }

  __device__ __host__ inline Vector3<T> operator>>(const T &rhs) const {
    Vector3<T> ret(*this);
    return ret >>= rhs;
  }

  __device__ __host__ inline bool operator==(const Vector3<T> &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  __device__ __host__ inline bool operator!=(const Vector3<T> &rhs) const {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }
};

template<typename T>
__device__ __host__ inline Vector3<T> operator+(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(rhs); 
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Vector3<T> operator-(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(rhs); 
  return ret -= lhs;
}

template<typename T>
__device__ __host__ inline Vector3<T> operator*(const T &lhs, const Vector3<T> &rhs) {
  Vector3<T> ret(rhs); 
  return ret *= lhs;
}

template<typename T>
class Vector4 {
 public:
  T x, y, z, w;
 public:
  __device__ __host__ Vector4<T>() {}

  __device__ __host__ Vector4<T>(const T &x_, const T &y_, const T &z_, const T &w_) 
      : x(x_), y(y_), z(z_), w(w_) {}

  __device__ __host__ explicit Vector4<T>(const Vector3<T> &others) 
      : x(others.x), y(others.y), z(others.z), w(1) {}

  __device__ __host__ static inline Vector4<T> Zeros() { return Vector4<T>({ 0, 0, 0, 0 }); }

  __device__ __host__ static inline Vector4<T> Ones() { return Vector4<T>({ 1, 1, 1, 1 }); }

  __device__ __host__ inline Vector4<T>& operator+=(const Vector4<T> &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator+=(const T &rhs) {
    x += rhs; y += rhs; z += rhs; w += rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator-=(const Vector4<T> &rhs) {
    x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator-=(const T &rhs) {
    x -= rhs; y -= rhs; z -= rhs; w -= rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T>& operator*=(const T &rhs) {
    x *= rhs; y *= rhs; z *= rhs; w *= rhs;
    return *this;
  }

  __device__ __host__ inline Vector4<T> operator+(const Vector4<T> &rhs) const {
    Vector4<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector4<T> operator+(const T &rhs) const {
    Vector4<T> ret(*this); 
    return ret += rhs;
  }

  __device__ __host__ inline Vector4<T> operator-(const Vector4<T> &rhs) const {
    Vector4<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector4<T> operator-(const T &rhs) const {
    Vector4<T> ret(*this); 
    return ret -= rhs;
  }

  __device__ __host__ inline Vector4<T> operator*(const T &rhs) const {
    Vector4<T> ret(*this);
    return ret *= rhs;
  }

  __device__ __host__ inline bool operator==(const Vector4<T> &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }

  __device__ __host__ inline bool operator!=(const Vector4<T> &rhs) const {
    return x != rhs.x || y != rhs.y || z != rhs.z || w != rhs.w;
  }
};

template<typename T>
__device__ __host__ inline Vector4<T> operator+(const T &lhs, const Vector4<T> &rhs) {
  Vector4<T> ret(rhs); 
  return ret += lhs;
}

template<typename T>
__device__ __host__ inline Vector4<T> operator-(const T &lhs, const Vector4<T> &rhs) {
  Vector4<T> ret(rhs); 
  return ret -= lhs;
}

