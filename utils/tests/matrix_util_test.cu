#include <gtest/gtest.h>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"

TEST(MatrixUtilTest, CameraIntrinsicsInverse) {
  const CameraIntrinsics<float> intrinsics(250, 250, 150, 150);
  const Matrix3<float> intrinsics_inv = intrinsics.Inverse();
  EXPECT_TRUE(intrinsics * intrinsics_inv == Matrix3<float>::Identity());
}

TEST(MatrixUtilTest, SO3Inverse) {
  const SO3<float> r(0, 1, 0, 0, 0, 1, 1, 0, 0);
  const SO3<float> r_inv = r.Inverse();
  EXPECT_TRUE(r * r_inv == Matrix3<float>::Identity());
  EXPECT_TRUE(r_inv * r == Matrix3<float>::Identity());
}

TEST(MatrixUtilTest, SE3Inverse) {
  const SO3<float> r(0, 1, 0, 0, 0, 1, 1, 0, 0);
  const Vector3<float> t(1, 2, 3);
  const SE3<float> transform(r, t);
  const SE3<float> transform_inv = transform.Inverse();
  EXPECT_TRUE(transform * transform_inv == Matrix4<float>::Identity());
  EXPECT_TRUE(transform_inv * transform == Matrix4<float>::Identity());
}
