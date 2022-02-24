/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 * See: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Embedding.cu
 */

/*!
 * \file src/op/dispatch/cuda/kernels/embedding_dx_cuda.cu
 * \brief embedding backward cuda kernel
 */
#include <stdio.h>
#include "./kernel_util.cuh"
namespace raf {
namespace op {
namespace cuda {

template void embedding_dense_backward_cuda<float, float>(const float*, float*, const int64_t*, int,
                                                          int, int, void*, int64_t);
template void embedding_dense_backward_cuda<__half, __half>(const __half*, __half*, const int64_t*,
                                                            int, int, int, void*, int64_t);
template void embedding_dense_backward_cuda<__half2, __half2>(const __half2*, __half2*,
                                                              const int64_t*, int, int, int, void*,
                                                              int64_t);

__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff) {
  return __ballot_sync(mask, predicate);
}

__host__ __device__ __forceinline__ int CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ float add(float a, float b) {
  return a + b;
}

__device__ __forceinline__ __half add(__half a, __half b) {
  return __hadd(a, b);
}

__device__ __forceinline__ __half2 add(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

static const int WARP_SIZE = 32;
static const int BLOCKDIMY = 32;

/*! \brief The kernel to perform embedding backward.
 * \param indices The input IDs.
 * \param grad The gradient from forward output (dy).
 * \param output The computed gradient by this kernel (dx).
 * \param n The size of indices.
 * \param range The maximum allowed value in indices.
 * \param stride The stride. It is usually the size of hidden stauts.
 */
template <typename scalar_t, typename accscalar_t>
__global__ void embedding_backward_feature_kernel(const int64_t* indices,
                                                  const scalar_t* __restrict__ grad,
                                                  accscalar_t* __restrict__ output, int n,
                                                  int range, int stride) {
  extern __shared__ char buf[];
  accscalar_t* smem = (accscalar_t*)buf;
  accscalar_t* my_s = smem + WARP_SIZE * threadIdx.y;
  int* indices_batch = (int*)(buf + sizeof(accscalar_t) * WARP_SIZE * blockDim.y);

  const int f = threadIdx.x + blockIdx.x * blockDim.x;  // feature_dim

  for (int batch_start = 0; batch_start < n; batch_start += blockDim.x * blockDim.y) {
    // Entire block cooperates to load a batch of 1024 indices to process
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (batch_start + tid < n) {
      int value = (int)indices[batch_start + tid];
      if (value >= range) {
        printf("indices[%d] = %d is out of range (%d)\n", batch_start + tid, value, range);
        asm("trap;");
      }
      indices_batch[tid] = value;
    }

    int batch_end =
        batch_start + blockDim.x * blockDim.y < n ? batch_start + blockDim.x * blockDim.y : n;

    // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
    for (int chunk_start = batch_start; chunk_start < batch_end; chunk_start += blockDim.y) {
      // This does double duty:  it makes sure indices_batch is ready, and it makes sure
      // match-group leaders are done with their accumulates before other warps start loading
      // again.
      __syncthreads();

      int n_this_chunk =
          (batch_end - chunk_start) < blockDim.y ? (batch_end - chunk_start) : blockDim.y;

      int src_row = chunk_start + threadIdx.y;
      int dst_row = indices_batch[src_row - batch_start];  // This warp's target row in grad_weight

      // All warps load their smem segments with incoming grad data
      if (src_row < n && f < stride) {
        my_s[threadIdx.x] = static_cast<accscalar_t>(grad[src_row * stride + f]);
      }

      __syncthreads();

      // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
      // We need to check if any other warps pulled grad data targeting dst_row.
      // If so, we elect the first warp in each matching group as the leader.
      // Each leader warp serializes the accumulates targeting dst_row in shared memory,
      // then finishes by adding the accumulated buffer to dst_row in grad_weight.
      if (src_row < n)  // Per-warp exit condition, safe with ballot_sync
      {
        int match_found_this_thread =
            (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
        if (threadIdx.x >= n_this_chunk) match_found_this_thread = 0;
        unsigned int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffs(matchmask) - 1;

        if (threadIdx.y == first_remaining_peer)  // Nominate lowest-indexed warp as the leader
        {
          matchmask ^= (1 << first_remaining_peer);
          while (matchmask) {
            first_remaining_peer = __ffs(matchmask) - 1;
            my_s[threadIdx.x] =
                add(my_s[threadIdx.x], smem[threadIdx.x + WARP_SIZE * first_remaining_peer]);
            matchmask ^= (1 << first_remaining_peer);
          }
          if (f < stride) {
            output[dst_row * stride + f] =
                add(output[dst_row * stride + f], static_cast<scalar_t>(my_s[threadIdx.x]));
          }
        }
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
void embedding_dense_backward_cuda(const scalar_t* grad, accscalar_t* output,
                                   const int64_t* indices, int num, int range, int stride,
                                   void* stream, int64_t element) {
  dim3 grid(CeilDiv(stride, (int64_t)WARP_SIZE));
  dim3 block(WARP_SIZE, BLOCKDIMY);
  CUDA_CALL(cudaMemsetAsync(output, 0, element * sizeof(accscalar_t), static_cast<cudaStream_t>(stream)));
  embedding_backward_feature_kernel<scalar_t, accscalar_t>
      <<<grid, block,
         sizeof(accscalar_t) * WARP_SIZE * BLOCKDIMY + sizeof(int) * WARP_SIZE * BLOCKDIMY,
         static_cast<cudaStream_t>(stream)>>>(
          indices, grad, output, num, range, stride);
}

}  // namespace cuda
}  // namespace op
}  // namespace raf
