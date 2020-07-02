#pragma once

struct Voxel {
  float         sdf;
  unsigned char rbg[3];
  unsigned char weight;
};

struct HashEntry {
  int position[3];
  int offset;
  int ptr;
};

