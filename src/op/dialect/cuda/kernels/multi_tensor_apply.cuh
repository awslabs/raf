/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dispatch/cuda/kernels/multi_tensor_apply.cuh
 * \brief multi_tennsor header file
 */

// TODO(@zhen-jia): The code structure is mostly migrate from Apex, we should consider the
//                  copyright when opensource

#include <vector>
namespace raf{
namespace op {
namespace cuda {

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

template<int n> struct TensorListMetadata
{
  void* addresses[n][depth_to_max_tensors[n-1]];
  int sizes[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  int block_to_chunk[depth_to_max_blocks[n-1]]; 
  int start_tensor_this_launch;
};

template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(int chunk_size, T tl, U callable,
                                          ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, tl, args...);
}

template <int depth, typename T, typename U, typename... ArgTypes>
void multi_tensor_apply(int block_size, int chunk_size, const std::vector<T*>& tensor_lists,
                        const std::vector<int>& numels, void* stream, U callable,
                        ArgTypes... args) {
  int ntensors = numels.size();

  TensorListMetadata<depth> tl;

  tl.start_tensor_this_launch = 0;
  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for(int t = 0; t < ntensors; t++)
  {
    tl.sizes[loc_tensor_info] = numels[t];
    for (int d = 0; d < depth; d++) {
      tl.addresses[d][loc_tensor_info] = tensor_lists[d * ntensors + t];
    }
    loc_tensor_info++;

    int chunks_this_tensor = (numels[t] + chunk_size - 1)/chunk_size;

    for(int chunk = 0; chunk < chunks_this_tensor; chunk++)
    {
      tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tl.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                           chunk == chunks_this_tensor - 1);
      bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
      bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
      if(tensors_full || blocks_full || last_chunk)
      {
        multi_tensor_apply_kernel<<<loc_block_info, block_size, 0,
                                    static_cast<cudaStream_t>(stream)>>>(chunk_size, tl, callable,
                                                                         args...);

        loc_block_info = 0;
        if(chunk == chunks_this_tensor - 1)
        {
          loc_tensor_info = 0;
          tl.start_tensor_this_launch = t + 1;
        } else {
          tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
          tl.start_tensor_this_launch = t;
        }
      }
    }
  }
}
}  // namespace cuda
}  // namespace op
}  // namespace raf

