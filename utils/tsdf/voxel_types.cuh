#pragma once

#include "utils/cuda/vector.cuh"

struct Voxel {
  float                   tsdf;
  Vector3<unsigned char>  rgb;
  unsigned char           weight;
};


