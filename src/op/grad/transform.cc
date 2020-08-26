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

Array<Expr> ReverseGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.reverse");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& axis = call->args[1];
  return {Call(op_dx, {dy, axis})};
}

MNM_OP_GRAD("mnm.op.reverse", ReverseGrad);

Array<Expr> ReverseSequenceGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.reverse_sequence");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 4);
  const Expr& seq_length = call->args[1];
  const Expr& seq_axis = call->args[2];
  const Expr& batch_axis = call->args[3];
  return {Call(op_dx, {dy, seq_length, seq_axis, batch_axis})};
}

MNM_OP_GRAD("mnm.op.reverse_sequence", ReverseSequenceGrad);

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

Array<Expr> ExpandDimsGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.expand_dims", ExpandDimsGrad);

Array<Expr> ReshapeGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto reshape = Op::Get("mnm.op.reshape");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  return {Call(reshape, {dy, Call(shape, {call->args[0]})})};
}

MNM_OP_GRAD("mnm.op.reshape", ReshapeGrad);

Array<Expr> TakeGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.take_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_EQ(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& indices = call->args[1];
  const Expr& axis = call->args[2];
  return {Call(op_dx, {x, y, dy, indices, axis})};
}

MNM_OP_GRAD("mnm.op.take", TakeGrad);

Array<Expr> CastGrad(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_dx = Op::Get("mnm.op.cast_like");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  return {Call(op_dx, {dy, x})};
}

MNM_OP_GRAD("mnm.op.cast", CastGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
