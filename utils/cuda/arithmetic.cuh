#pragma once

#include <cassert>

#include "utils/cuda/errors.cuh"

#define SCAN_BLOCK_SIZE 1024
#define SCAN_PAD(x) (x + (x) / 32)

template<typename T>
__global__ void auxiliary_sum_kernel(T *input, T *output, T *aux, int len) {
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

template<typename T>
__global__ void scan_kernel(T *input, T *output, T *auxout, int len) {
  __shared__ T buffer[SCAN_PAD(2*SCAN_BLOCK_SIZE)];

  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int i1 = bx * SCAN_BLOCK_SIZE * 2 + tx;
  const int i2 = i1 + SCAN_BLOCK_SIZE;

  buffer[SCAN_PAD(tx)] = (i1 < len) ? input[i1] : 0;
  buffer[SCAN_PAD(tx+SCAN_BLOCK_SIZE)] = (i2 < len) ? input[i2] : 0;

  // pre scan
  for (int stride = 1; stride < 2*SCAN_BLOCK_SIZE; stride <<= 1) {
    __syncthreads();
    const int idx = (tx + 1) * stride * 2 - 1;
    if (idx < 2*SCAN_BLOCK_SIZE) {
      buffer[SCAN_PAD(idx)] += buffer[SCAN_PAD(idx-stride)];
    }
  }

  // post scan
  for (int stride = SCAN_BLOCK_SIZE/2; stride > 0; stride >>= 1) {
    const int idx = (tx + 1) * stride * 2 - 1;
    if (idx + stride < 2*SCAN_BLOCK_SIZE) {
      buffer[SCAN_PAD(idx+stride)] += buffer[SCAN_PAD(idx)];
    }
    __syncthreads();
  }

  if (i1 < len) output[i1] = buffer[SCAN_PAD(tx)];
  if (i2 < len) output[i2] = buffer[SCAN_PAD(tx+SCAN_BLOCK_SIZE)];

  if (tx == 0 && auxout) {
    auxout[bx] = buffer[SCAN_PAD(2*SCAN_BLOCK_SIZE-1)];
  }
}

template<typename T>
void prefix_sum(T *input, T *output, T *auxout, int len, cudaStream_t stream = NULL) {
  // cannot handle more than (1 << 22) elements
  assert(len < SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE * 4);

  const int num_aux = ceil((float)len / (2*SCAN_BLOCK_SIZE));

  scan_kernel<T><<<num_aux, SCAN_BLOCK_SIZE, 0, stream>>>(input, output, auxout, len);
  scan_kernel<T><<<1, SCAN_BLOCK_SIZE, 0, stream>>>(auxout, auxout, NULL, num_aux);
  auxiliary_sum_kernel<T><<<num_aux, SCAN_BLOCK_SIZE, 0, stream>>>(output, output, auxout, len);
  CUDA_CHECK_ERROR;
}
