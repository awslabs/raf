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

template<bool transpose_a, bool transpose_b>
Array<Expr> MatmulGradImpl(const Var& y, const Expr& orig_call, const Array<Expr>& ograds) {
  CHECK_EQ(ograds.size(), 1);
  const Expr& dy = ograds[0];
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];

  Call da;
  if (!transpose_a) {
    // matmul(dy, false, b, !transpose_b);
    static auto op = Op::Get(!transpose_b ? "mnm.op.matmul_nt" : "mnm.op.matmul");
    da = CallNode::make(op, {dy, b});
  } else {
    // matmul(b, transpose_b, dy, true);
    static auto op = Op::Get(transpose_b ? "mnm.op.matmul_tt" : "mnm.op.matmul_nt");
    da = CallNode::make(op, {b, dy});
  }

  Call db;
  if (!transpose_b) {
    // matmul(a, !transpose_a, dy, false);
    static auto op = Op::Get(!transpose_a ? "mnm.op.matmul_tn" : "mnm.op.matmul");
    db = CallNode::make(op, {a, dy});
  } else {
    // matmul(dy, true, a, transpose_a);
    static auto op = Op::Get(transpose_a ? "mnm.op.matmul_tt" : "mnm.op.matmul_tn");
    db = CallNode::make(op, {dy, a});
  }

  return {da, db};
}

auto MatmulGradNN = MatmulGradImpl<false, false>;
auto MatmulGradNT = MatmulGradImpl<false, true>;
auto MatmulGradTN = MatmulGradImpl<true, false>;
auto MatmulGradTT = MatmulGradImpl<true, true>;

MNM_OP_GRAD("mnm.op.matmul", MatmulGradNN);
MNM_OP_GRAD("mnm.op.matmul_nt", MatmulGradNT);
MNM_OP_GRAD("mnm.op.matmul_tn", MatmulGradTN);
MNM_OP_GRAD("mnm.op.matmul_tt", MatmulGradNN);

}  // namespace grad
}  // namespace op
}  // namespace mnm
