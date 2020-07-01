#pragma once

/* vector types to be consistent with CUDA */
struct float2 {
  float x;
  float y;
};

struct float3 {
  float x;
  float y;
  float z;
};

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

struct int2 {
  int x;
  int y;
};

struct int3 {
  int x;
  int y;
  int z;
};

struct int4 {
  int x;
  int y;
  int z;
  int w;
};

struct uint2 {
  unsigned int x;
  unsigned int y;
};

struct uint3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct uint4 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
};
