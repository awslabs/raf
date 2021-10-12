/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dispatch/cuda/kernel/kernel_utils.h
 * \brief Helper functions for CUDA kernels
 */
#pragma once
#include <cuda_fp16.h>
namespace mnm {
namespace op {
namespace cuda {

template <typename scalar_t, typename accscalar_t>
void embedding_dense_backward_cuda(const scalar_t* grad, accscalar_t* output,
                                   const int64_t* indices, int num, int range,
                                   int stride, void* stream, int64_t element);

}  // namespace cuda
}  // namespace op
}  // namespace mnm

