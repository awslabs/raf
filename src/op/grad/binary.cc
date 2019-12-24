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

Array<Expr> AddGrad(const Var& y, const Expr& orig_call, const Array<Expr>& ograds) {
  // schema for relu is:
  //    x1, x2
  // schema for binary_dx is:
  //    x1, x2, y, dy
  static auto op_dx = Op::Get("mnm.op.add_dx");
  CHECK_EQ(ograds.size(), 1);
  const Expr& dy = ograds[0];
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  return {
      CallNode::make(op_dx, {x1, x2, y, dy}),
      CallNode::make(op_dx, {x2, x1, y, dy}),
  };
}

MNM_OP_GRAD("mnm.op.add", AddGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
