/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/cuda_utils.cc
 * \brief CUDA dialect utils
 */
#include "raf/op.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace cuda {

RAF_REGISTER_DIALECT("cuda").set_enable(DevType::kCUDA());

}  // namespace cuda
}  // namespace op
}  // namespace raf
