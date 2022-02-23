/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/cuda_utils.cc
 * \brief CUDA dialect utils
 */
#include "mnm/op.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace cuda {

MNM_REGISTER_DIALECT("cuda").set_enable(DevType::kCUDA());

}  // namespace cuda
}  // namespace op
}  // namespace mnm
