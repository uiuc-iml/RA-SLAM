#pragma once

#ifndef NDEBUG
#include <cstdio>
#endif

#define CUDA_SAFE_CALL(err) cuda_safe_call(err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR cuda_check_error(__FILE__, __LINE__)

__device__ __host__ inline void cuda_safe_call(const cudaError_t err, const char *file, int line) {
#ifndef NDEBUG
  if (err != cudaSuccess) {
    const char *err_msg = cudaGetErrorString(err); 
    printf("\033[0;31mCUDA Error: %s at %s:%d\n\033[0m", err_msg, file, line);
  }
#endif
}

__device__ __host__ inline void cuda_check_error(const char *file, int line) {
#ifndef NDEBUG
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    const char *err_msg = cudaGetErrorString(err);
    printf("\033[0;31mCUDA Error: %s at %s:%d\n\033[0m", err_msg, file, line);
  }
#endif
}
