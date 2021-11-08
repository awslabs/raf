/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/grad/unary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

MNM_OP_GRAD("mnm.op.numel", NoGrads<1>);
MNM_OP_GRAD("mnm.op.shape_as_tensor", NoGrads<1>);

}  // namespace grad
}  // namespace op
}  // namespace mnm
