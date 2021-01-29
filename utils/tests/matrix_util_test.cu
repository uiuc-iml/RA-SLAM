#include <gtest/gtest.h>

#include "utils/cuda/camera.cuh"
#include "utils/cuda/lie_group.cuh"

TEST(MatrixUtilTest, CameraIntrinsicsInverse) {
  const CameraIntrinsics<float> intrinsics(250, 250, 150, 150);
  const Matrix3<float> intrinsics_inv = intrinsics.Inverse();
  EXPECT_TRUE(static_cast<Matrix3<float>>(intrinsics) * intrinsics_inv ==
              Matrix3<float>::Identity());
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
  const SE3<float> pose(r, t);
  const SE3<float> pose_inv = pose.Inverse();
  EXPECT_TRUE(pose * pose_inv == Matrix4<float>::Identity());
  EXPECT_TRUE(pose_inv * pose == Matrix4<float>::Identity());
}

TEST(MatrixUtilTest, SE3Apply) {
  const Vector3<float> vec3(3, 7, 21);
  const Vector4<float> vec3_h(vec3);
  const SO3<float> r(0, 1, 0, 0, 0, 1, 1, 0, 0);
  const Vector3<float> t(1, 2, 3);
  const SE3<float> pose(r, t);

  const Vector3<float> ret1 = pose.Apply(vec3);
  const Vector4<float> ret2_h = pose * vec3_h;

  EXPECT_TRUE(Vector4<float>(ret1) == ret2_h);
}
