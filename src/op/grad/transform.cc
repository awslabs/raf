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

Array<Expr> BatchFlattenGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.batch_flatten", BatchFlattenGrad);

Array<Expr> TransposeGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto transpose_dx = Op::Get("mnm.op.transpose_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const Expr& axes = call->args[1];
  return {Call(transpose_dx, {x, y, dy, axes})};
}

MNM_OP_GRAD("mnm.op.transpose", TransposeGrad);

Array<Expr> ConcatenateGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.split");
  static auto op_indices = Op::Get("mnm.op.concatenate_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  Expr indices = Call(op_indices, {x, axis});
  return {Call(op_dx, {dy, indices, axis})};
}

MNM_OP_GRAD("mnm.op.concatenate", ConcatenateGrad);

Array<Expr> ClipGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.clip_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& a_min = call->args[1];
  const Expr& a_max = call->args[2];
  return {Call(op_dx, {x, dy, a_min, a_max})};
}

MNM_OP_GRAD("mnm.op.clip", ClipGrad);

Array<Expr> ReshapeGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.reshape");
  static auto op_shape = Op::Get("mnm.op.reshape_dx");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& orig_shape = call->args[1];
  Expr shape = Call(op_shape, {x, orig_shape});
  return {Call(op_dx, {dy, shape})};
}

MNM_OP_GRAD("mnm.op.reshape", ReshapeGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
