/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification BSD 3-Clause
 * See: https://github.com/szhengac/apex/blob/lans/csrc/multi_tensor_lans.cu
 */

/*!
 * \file src/op/dispatch/cuda/kernels/multi_tensor_lans.cu
 * \brief lans cuda kernel
 */
#include "./kernel_util.cuh"
#include "./multi_tensor_apply.cuh"
#define BLOCK_SIZE 512
#define ILP 4

namespace raf {
namespace op {
namespace cuda {

template void multi_tensor_lans_cuda<float>(
    int chunk_size, std::vector<float*> tensor_lists, const float lr, const float beta1,
    const float beta2, const float epsilon, const int bias_correction, const float bias_correction1,
    const float bias_correction2, const float beta3, const float weight_decay,
    const int grad_averaging, const int mode, const bool normalize_grad,
    const std::vector<int> numels, void* stream, float* output_per_tensor, float* grad_norm_tensor,
    float* param_norm_tensor, float* update_m_norm, float* q_norm_tensor,
    int max_chunks_per_tensor);

typedef enum {
  MOMENT_MODE_0 = 0,  // L2 regularization mode
  MOMENT_MODE_1 = 1   // Decoupled weight decay mode
} adamMode_t;

using MATH_T = float;
template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes(T* x, T val, int lanes = 1,
                        bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x*blockDim.y; // blockSize is intended to be a multiple of 32.

  if(blockSize >= 64)
  {
    x[tid] = val;
    __syncthreads();
  }

  #pragma unroll
  for(int i = (blockSize >> 1); i >= 64; i >>= 1)
  {
    if(tid < i)
      x[tid] = x[tid] + x[tid+i];
    __syncthreads();
  }

  T final;

  if(tid < 32)
  {
    if(blockSize >= 64)
      final = x[tid] + x[tid+32];
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for(int i = 16; i >= lanes; i >>= 1)
      final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if(share_result)
  {
    if(tid < lanes)
      x[tid] = final; // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template<typename x_t>
struct L2NormFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<1>& tl,
    float* output_per_tensor,
    int max_chunks_per_tensor)
  {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP]; // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for(int i = 0; i < ILP; i++)
    {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x))
    {
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_x, x, 0 , i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next*next;
        }
      }
    }
    else
    {
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
      {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            float next = static_cast<float>(x[i]);
            vals[ii] += next*next;
          }
        }
      }
    }

    float val = 0.f;
    for(int i = 0; i < ILP; i++)
        val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if(threadIdx.x == 0)
    {
      output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
                        chunk_idx] = final;
    }
  }
};

__global__ void cleanup(
  float* output_per_tensor,
  float* ret_per_tensor,
  int max_chunks_per_tensor)
{
  __shared__ float vals[512];

  float* output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

  float val = 0;
  for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
    val += output_this_tensor[i];

  float final = reduce_block_into_lanes(vals, val);

  if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
}

template <typename T>
void multi_tensor_l2norm_cuda(int chunk_size, std::vector<T*> tensor_lists,
                              std ::vector<int> numels, float* output_per_tensor,
                              float* ret_per_tensor, void* stream,
                              int max_chunks_per_tensor) {

  int ntensors = numels.size();

  CUDA_CALL(cudaMemsetAsync(output_per_tensor, 0, ntensors * max_chunks_per_tensor * sizeof(float),
                            static_cast<cudaStream_t>(stream)));
  multi_tensor_apply<1>(BLOCK_SIZE, chunk_size, tensor_lists, numels, stream,
                        L2NormFunctor<T>(), output_per_tensor, max_chunks_per_tensor);

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  cleanup<<<ntensors, 512, 0, static_cast<cudaStream_t>(stream)>>>(
      output_per_tensor, ret_per_tensor, max_chunks_per_tensor);
}

template<typename T>
struct LANSStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<5>& tl,
    const float beta1,
    const float beta2,
    const float beta3,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    adamMode_t mode,
    const float decay,
    float* per_tensor_grad_norm,
    bool normalize_grad)
  {
    // I'd like this kernel to propagate infs/nans.
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float grad_norm = per_tensor_grad_norm[tensor_num];
    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;
   
	T* q = (T*)tl.addresses[1][tensor_loc];
    q += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[2][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[3][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[4][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_q[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_q[ii] = q[i];
          // special ?optimization? for lans stage 1
          if (decay == 0) {
            r_p[ii] = MATH_T(0);
          }
          else {
            r_p[ii] = p[i];
          }
          r_m[ii] = m[i];
          r_v[ii] = v[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_q[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        MATH_T scaled_grad = r_g[ii];
        if (normalize_grad && grad_norm != 0.0f) {
           scaled_grad /= (grad_norm + epsilon);
        }
        if (mode == MOMENT_MODE_0) {
          // L2 on scaled grad
          scaled_grad = scaled_grad + decay*r_p[ii];
          r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
          r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          r_p[ii] = next_m_unbiased / denom;
          r_q[ii] = scaled_grad / denom;
        }
        else {
          r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
          r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T scaled_p = decay * r_p[ii];
          r_p[ii] = (next_m_unbiased/denom) + scaled_p;
          r_q[ii] = (scaled_grad/denom) + scaled_p;
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if (i < n && i < chunk_size) {
          g[i] = r_p[ii];
          q[i] = r_q[ii];
          m[i] = r_m[ii];
          v[i] = r_v[ii];
        }
      }
    }
  }
};

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
template<typename T>
struct LANSStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta3,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_m_norm,
    const float* per_tensor_update_g_norm,
    const float learning_rate)
  {
    // I'd like this kernel to propagate infs/nans.

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float param_norm = per_tensor_param_norm[tensor_num];
    float update_m_norm = per_tensor_update_m_norm[tensor_num];
    float update_g_norm = per_tensor_update_g_norm[tensor_num];
    MATH_T ratio_m = (update_m_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_m_norm) : learning_rate;
    MATH_T ratio_g = (update_g_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_g_norm) : learning_rate;
    ratio_m *= beta1;
    ratio_g *= beta3;

    T* update_m = (T*)tl.addresses[0][tensor_loc];
    update_m += chunk_idx*chunk_size;

    T* update_g = (T*)tl.addresses[1][tensor_loc];
    update_g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[2][tensor_loc];
    p += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_p[ILP];
      MATH_T r_update_m[ILP];
      MATH_T r_update_g[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
       	int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_update_m[ii] = update_m[i];
          r_update_g[ii] = update_g[i];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        r_p[ii] = r_p[ii] - (ratio_m * r_update_m[ii]) - (ratio_g * r_update_g[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
        }
      }
    }
  }
};

template <typename T>
void multi_tensor_lans_cuda(int chunk_size, std::vector<T*> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int bias_correction, const float bias_correction1,
                            const float bias_correction2, const float beta3,
                            const float weight_decay, const int grad_averaging, const int mode,
                            const bool normalize_grad, const std::vector<int> numels, void* stream,
                            float* output_per_tensor, float* grad_norm_tensor,
                            float* param_norm_tensor, float* update_m_norm, float* q_norm_tensor,
                            int max_chunks_per_tensor) {
  // Master weight and 32bit momentum(potentially changing) is not handled by this
  // So we assume every tensor are all in the same type

  int param_group_n = numels.size();
  std::vector<T*> grad_list(tensor_lists.begin(), tensor_lists.begin() + param_group_n);
  std::vector<T*> param_list(tensor_lists.begin() + 2 * param_group_n,
                             tensor_lists.begin() + 3 * param_group_n);

  // Compute per-layer grad norm
  multi_tensor_l2norm_cuda(chunk_size, grad_list, numels, output_per_tensor, grad_norm_tensor,
                           stream, max_chunks_per_tensor);

  // Compute per tensor param norm
  multi_tensor_l2norm_cuda(chunk_size, param_list, numels, output_per_tensor, param_norm_tensor,
                           stream, max_chunks_per_tensor);

  multi_tensor_apply<5>(BLOCK_SIZE, chunk_size, tensor_lists, numels, stream,
                        LANSStage1Functor<float>(), beta1, beta2,
                        beta3,  // 1-beta1 or 1 depends on averaging mode
                        bias_correction1, bias_correction2, epsilon, (adamMode_t)mode, weight_decay,
                        grad_norm_tensor, normalize_grad);

  // Compute update norms
  multi_tensor_l2norm_cuda(chunk_size, grad_list, numels, output_per_tensor, update_m_norm, stream,
                           max_chunks_per_tensor);

  std::vector<T*> q_list(tensor_lists.begin() + param_group_n,
                         tensor_lists.begin() + 2 * param_group_n);

  multi_tensor_l2norm_cuda(chunk_size, q_list, numels, output_per_tensor, q_norm_tensor, stream,
                           max_chunks_per_tensor);

  std::vector<T*> grad_q_param_list(tensor_lists.begin(), tensor_lists.begin()+3* param_group_n);

  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, grad_q_param_list, numels, stream,
                        LANSStage2Functor<float>(), beta1, beta3, param_norm_tensor, update_m_norm,
                        q_norm_tensor, lr);

  return;
}

}  // namespace cuda
}  // namespace op
}  // namespace raf

