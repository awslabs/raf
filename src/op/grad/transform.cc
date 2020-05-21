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

Array<Expr> BatchFlattenGrad(const Expr& orig_call, const Var &y, const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {CallNode::make(reshape, {dy, CallNode::make(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.batch_flatten", BatchFlattenGrad);

Array<Expr> TransposeGrad(const Expr& orig_call, const Var &y, const Expr& dy) {
  static auto transpose_dx = Op::Get("mnm.op.transpose_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& axes = call->args[1];
  return {CallNode::make(transpose_dx, {x, y, dy, axes})};
}

MNM_OP_GRAD("mnm.op.transpose", TransposeGrad);

Array<Expr> ConcatenateGrad(const Expr& orig_call, const Var &y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.split");
  static auto op_indices = Op::Get("mnm.op.concatenate_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr indices = CallNode::make(op_indices, {x, axis});
  return {CallNode::make(op_dx, {dy, indices, axis})};
}

MNM_OP_GRAD("mnm.op.concatenate", ConcatenateGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
