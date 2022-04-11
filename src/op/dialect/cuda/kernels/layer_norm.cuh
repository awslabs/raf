/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dispatch/cuda/kernel/layer_norm.cuh
 * \brief Headers of CUDA layer_norm and layer_norm_dx kernels
 */
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <vector>
#include "./Half.h"
#include "../../../../common/cuda_utils.h"


namespace raf {
namespace op {
namespace cuda {
template <typename T, typename U, typename V>
void HostApplyLayerNorm(V* output, U* mean, U* invvar, const T* input, int n1, int n2,
                        const T* gamma, const T* beta, double epsilon, void* stream,
                        const uint64_t maxGridY);

template <typename T, typename V>
void HostLayerNormGradient(const V* dout, const float* mean, const float* invvar, T* input, int n1,
                           int n2, const V* gamma, double epsilon, T* grad_input, V* grad_gamma,
                           V* grad_beta, float* part_gard_gamma, float* part_grad_beta,
                           void* stream, const uint64_t maxGridY);

}  // namespace cuda
}  // namespace op
}  // namespace raf
