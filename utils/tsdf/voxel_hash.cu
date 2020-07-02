#include "voxel_hash.h"

#define NUM_BUCKET 0x100000

__host__ __device__ int hash(const short3 &position) {
  return position.x; 
}

__host__ __device__ HashEntry* retrieve(const HashEntry *hash_table, const short3 &position) {
  
}

