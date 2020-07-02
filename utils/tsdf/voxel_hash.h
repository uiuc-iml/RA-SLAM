#pragma once

#include "utils/tsdf/voxel_types.h"

__host__ __device__ int hash(const short3 &position);

__host__ __device__ HashEntry* retrieve(const HashEntry *hash_table, const short3 &position);
