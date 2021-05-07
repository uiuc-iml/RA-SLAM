#pragma once

#include <cuda_runtime.h>

#include <cassert>

#include "utils/cuda/errors.cuh"

#define SCAN_BLOCK_SIZE 1024
#define SCAN_PAD(x) (x + (x) / 32)

/**
 * @brief accumulate auxiliary data back into the scanned output
 *
 * @tparam T      data type
 * @param input   pointer to GPU input buffer
 * @param output  pointer to GPU output buffer
 * @param aux     pointer to GPU auxiliary buffer
 * @param len     length of the input / output array
 */
template <typename T>
__global__ void auxiliary_sum_kernel(T* input, T* output, T* aux, int len) {
  __shared__ T aux_offset;

  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int i1 = (bx + 1) * SCAN_BLOCK_SIZE * 2 + tx;
  const int i2 = i1 + SCAN_BLOCK_SIZE;

  if (tx == 0) {
    aux_offset = aux[bx];
  }

  __syncthreads();

  if (i1 < len) output[i1] = input[i1] + aux_offset;
  if (i2 < len) output[i2] = input[i2] + aux_offset;
}

/**
 * @brief perform parallel prefix-sum using a work efficient algorithm described
 * in
 *        https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 *
 * @tparam T      data type
 * @param input   pointer to GPU input buffer
 * @param output  pointer to GPU output buffer
 * @param auxout  optional pointer to GPU auxiliary buffer
 * @param len     length of input / output array
 */
template <typename T>
__global__ void scan_kernel(T* input, T* output, T* auxout, int len) {
  __shared__ T buffer[SCAN_PAD(2 * SCAN_BLOCK_SIZE)];

  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int i1 = bx * SCAN_BLOCK_SIZE * 2 + tx;
  const int i2 = i1 + SCAN_BLOCK_SIZE;

  buffer[SCAN_PAD(tx)] = (i1 < len) ? input[i1] : 0;
  buffer[SCAN_PAD(tx + SCAN_BLOCK_SIZE)] = (i2 < len) ? input[i2] : 0;

  // pre scan
  for (int stride = 1; stride < 2 * SCAN_BLOCK_SIZE; stride <<= 1) {
    __syncthreads();
    const int idx = (tx + 1) * stride * 2 - 1;
    if (idx < 2 * SCAN_BLOCK_SIZE) {
      buffer[SCAN_PAD(idx)] += buffer[SCAN_PAD(idx - stride)];
    }
  }

  // post scan
  for (int stride = SCAN_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    const int idx = (tx + 1) * stride * 2 - 1;
    if (idx + stride < 2 * SCAN_BLOCK_SIZE) {
      buffer[SCAN_PAD(idx + stride)] += buffer[SCAN_PAD(idx)];
    }
    __syncthreads();
  }

  if (i1 < len) output[i1] = buffer[SCAN_PAD(tx)];
  if (i2 < len) output[i2] = buffer[SCAN_PAD(tx + SCAN_BLOCK_SIZE)];

  if (tx == 0 && auxout) {
    auxout[bx] = buffer[SCAN_PAD(2 * SCAN_BLOCK_SIZE - 1)];
  }
}

/**
 * @brief CPU wrapper for a one-level GPU parallel sum kernel
 *        [for length up to (SCAN_BLOCK_SIZE * 2)^2]
 *
 * @tparam T      data type
 * @param input   pointer to GPU input buffer
 * @param output  pointer to GPU output buffer
 * @param auxout  pointer to GPU auxiliary buffer
 * @param len     length of input / ouput array
 * @param stream  optional CUDA stream
 */
template <typename T>
void prefix_sum_1(T* input, T* output, T* auxout, int len, cudaStream_t stream = NULL) {
  // cannot handle more than (1 << 22) elements
  assert(len <= SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE * 4);

  const int num_aux = ceil((float)len / (2 * SCAN_BLOCK_SIZE));

  // allocate if necessary
  T* aux_tmp = auxout;
  if (!auxout) {
    CUDA_SAFE_CALL(cudaMalloc(&aux_tmp, sizeof(T) * num_aux));
  }

  scan_kernel<T><<<num_aux, SCAN_BLOCK_SIZE, 0, stream>>>(input, output, aux_tmp, len);
  CUDA_STREAM_CHECK_ERROR(stream);
  scan_kernel<T><<<1, SCAN_BLOCK_SIZE, 0, stream>>>(aux_tmp, aux_tmp, NULL, num_aux);
  CUDA_STREAM_CHECK_ERROR(stream);

  auxiliary_sum_kernel<T><<<num_aux, SCAN_BLOCK_SIZE, 0, stream>>>(output, output, aux_tmp, len);
  CUDA_STREAM_CHECK_ERROR(stream);

  // deallocate if necessary
  if (!auxout) {
    CUDA_SAFE_CALL(cudaFree(aux_tmp));
  }
}

/**
 * @brief CPU wrapper for a GPU parallel sum kernel [for length up to (SCAN_BLOCK_SIZE * 2)^3]
 *
 * @tparam T      data type
 * @param input   pointer to GPU input buffer
 * @param output  pointer to GPU output buffer
 * @param auxout  pointer to GPU auxiliary buffer for all levels
 * @param len     length of input / ouput array
 * @param stream  optional CUDA stream
 */
template <typename T>
void prefix_sum(T* input, T* output, T* auxout, int len, cudaStream_t stream = NULL) {
  const int num_aux1 = ceil((float)len / (2 * SCAN_BLOCK_SIZE));
  const int num_aux2 = ceil((float)num_aux1 / (2 * SCAN_BLOCK_SIZE));

  if (num_aux2 <= 1) {
    prefix_sum_1<T>(input, output, auxout, len, stream);
    return;
  }

  // allocate if necessary
  T* aux1 = auxout;
  if (!auxout) {
    CUDA_SAFE_CALL(cudaMalloc(&aux1, sizeof(T) * (num_aux1 + num_aux2)));
  }
  T* aux2 = aux1 + num_aux1;

  scan_kernel<T><<<num_aux1, SCAN_BLOCK_SIZE, 0, stream>>>(input, output, aux1, len);
  CUDA_STREAM_CHECK_ERROR(stream);
  scan_kernel<T><<<num_aux2, SCAN_BLOCK_SIZE, 0, stream>>>(aux1, aux1, aux2, num_aux1);
  CUDA_STREAM_CHECK_ERROR(stream);
  scan_kernel<T><<<1, SCAN_BLOCK_SIZE, 0, stream>>>(aux2, aux2, nullptr, num_aux2);
  CUDA_STREAM_CHECK_ERROR(stream);

  auxiliary_sum_kernel<T><<<num_aux2, SCAN_BLOCK_SIZE, 0, stream>>>(aux1, aux1, aux2, num_aux1);
  CUDA_STREAM_CHECK_ERROR(stream);
  auxiliary_sum_kernel<T><<<num_aux1, SCAN_BLOCK_SIZE, 0, stream>>>(output, output, aux1, len);
  CUDA_STREAM_CHECK_ERROR(stream);

  // deallocate if necessary
  if (!auxout) {
    CUDA_SAFE_CALL(cudaFree(aux1));
  }
}
