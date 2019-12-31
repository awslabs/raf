/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "mnm/op.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> BatchFlattenGrad(const Var& y, const Expr& orig_call, const Array<Expr>& ograds) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  CHECK_EQ(ograds.size(), 1);
  const Expr& dy = ograds[0];
  const CallNode* call = orig_call.as<CallNode>();
  return {CallNode::make(reshape, {dy, CallNode::make(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.batch_flatten", BatchFlattenGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
