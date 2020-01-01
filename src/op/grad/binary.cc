/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> AddGrad(const Expr& orig_call, const Var &y, const Expr& dy) {
  // schema for relu is:
  //    x1, x2
  // schema for binary_dx is:
  //    x1, x2, y, dy
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];

  auto f = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = CallNode::make(collapse_axis, {dy, x});
    Call keep = CallNode::make(collapse_keep, {dy, x});
    return CallNode::make(sum, {dy, axes, keep});
  };

  return {f(x1), f(x2)};
}

MNM_OP_GRAD("mnm.op.add", AddGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
