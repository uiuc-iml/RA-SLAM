#pragma once

#ifndef NDEBUG
#include <cstdio>
#endif

#define CUDA_SAFE_CALL(err) cuda_safe_call(err, __FILE__, __LINE__)

#define CUDA_SAFE_DEVICE_SYNC cuda_safe_call(cudaDeviceSynchronize(), __FILE__, __LINE__)

#define CUDA_CHECK_ERROR cuda_check_error(__FILE__, __LINE__)

inline void cuda_safe_call(const cudaError_t err, const char *file, int line) {
#ifndef NDEBUG
  if (err != cudaSuccess) {
    const char *err_msg = cudaGetErrorString(err); 
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", err_msg, file, line);
  }
#endif
}

inline void cuda_check_error(const char *file, int line) {
#ifndef NDEBUG
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    const char *err_msg = cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", err_msg, file, line);
  }
#endif
}
