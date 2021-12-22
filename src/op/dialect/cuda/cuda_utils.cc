/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dialect/cuda/cuda_utils.cc
 * \brief CUDA dialect utils
 */
#include "mnm/op.h"
namespace mnm {
namespace op {
namespace cuda {

MNM_REGISTER_DIALECT("cuda").set_enable(DevType::kCUDA());

}  // namespace cuda
}  // namespace op
}  // namespace mnm
