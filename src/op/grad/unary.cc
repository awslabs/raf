/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/unary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

RAF_OP_GRAD("raf.op.numel", NoGrads<1>);
RAF_OP_GRAD("raf.op.shape_as_tensor", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace raf
