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

template <bool transpose_a, bool transpose_b>
Array<Expr> MatmulGradImpl(const Expr& orig_call, const Var& y, const Expr& dy) {
  static auto op_nn = Op::Get("mnm.op.matmul");
  static auto op_nt = Op::Get("mnm.op.matmul_nt");
  static auto op_tn = Op::Get("mnm.op.matmul_tn");
  static auto op_tt = Op::Get("mnm.op.matmul_tt");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];
  Call da, db;
  if (!transpose_a) {
    if (!transpose_b) {
      return {
          CallNode::make(op_nt, {dy, b}),
          CallNode::make(op_tn, {a, dy}),
      };
    } else {
      return {
          da = CallNode::make(op_nn, {dy, b}),
          db = CallNode::make(op_tn, {dy, a}),
      };
    }
  } else {
    if (!transpose_b) {
      return {
          CallNode::make(op_nt, {b, dy}),
          CallNode::make(op_nn, {a, dy}),
      };
    } else {
      return {
          CallNode::make(op_tt, {b, dy}),
          CallNode::make(op_tt, {dy, a}),
      };
    }
  }
  LOG(FATAL) << "Unreachable code";
  throw;
}

auto MatmulGradNN = MatmulGradImpl<false, false>;
auto MatmulGradNT = MatmulGradImpl<false, true>;
auto MatmulGradTN = MatmulGradImpl<true, false>;
auto MatmulGradTT = MatmulGradImpl<true, true>;

MNM_OP_GRAD("mnm.op.matmul", MatmulGradNN);
MNM_OP_GRAD("mnm.op.matmul_nt", MatmulGradNT);
MNM_OP_GRAD("mnm.op.matmul_tn", MatmulGradTN);
MNM_OP_GRAD("mnm.op.matmul_tt", MatmulGradTT);

}  // namespace grad
}  // namespace op
}  // namespace mnm
