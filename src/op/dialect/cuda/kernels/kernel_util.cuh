/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/op/dispatch/cuda/kernel/kernel_utils.h
 * \brief Helper functions for CUDA kernels
 */
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <vector>
#include "../../../../common/cuda_utils.h"

namespace mnm {
namespace op {
namespace cuda {

template <typename scalar_t, typename accscalar_t>
void embedding_dense_backward_cuda(const scalar_t* grad, accscalar_t* output,
                                   const int64_t* indices, int num, int range,
                                   int stride, void* stream, int64_t element);

template <typename T>
void multi_tensor_lans_cuda(int chunk_size, std::vector<T*> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int bias_correction, const float bias_correction1,
                            const float bias_correction2, const float beta3,
                            const float weight_decay, const int grad_averaging, const int mode,
                            const bool normalize_grad, const std::vector<int> numels, void* stream,
                            float* output_per_tensor, float* grad_norm_tensor,
                            float* param_norm_tensor, float* update_m_norm, float* q_norm_tensor,
                            int max_chunks_per_tensor);

}  // namespace cuda
}  // namespace op
}  // namespace mnm

