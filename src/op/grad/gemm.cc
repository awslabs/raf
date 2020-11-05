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
using namespace mnm::value;

template <bool transpose_a, bool transpose_b>
Array<Expr> MatmulGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
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
          Call(op_nt, {dy, b}),
          Call(op_tn, {a, dy}),
      };
    } else {
      return {
          da = Call(op_nn, {dy, b}),
          db = Call(op_tn, {dy, a}),
      };
    }
  } else {
    if (!transpose_b) {
      return {
          Call(op_nt, {b, dy}),
          Call(op_nn, {a, dy}),
      };
    } else {
      return {
          Call(op_tt, {b, dy}),
          Call(op_tt, {dy, a}),
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

Array<Expr> BatchMatmulGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto batch_matmul = Op::Get("mnm.op.batch_matmul");
  static auto transpose = Op::Get("mnm.op.transpose");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];
  const std::vector<Value> axes = {IntValue::make(0), IntValue::make(2), IntValue::make(1)};
  const Expr& axes_expr = MakeConstant(TupleValue::make(Array<Value>(axes)));
  auto dy_trans = Call(transpose, {dy, axes_expr});
  auto b_trans = Call(transpose, {b, axes_expr});
  auto a_trans = Call(transpose, {a, axes_expr});
  return {
      Call(batch_matmul, {dy, b_trans}),
      Call(batch_matmul, {dy_trans, a_trans}),
  };
}

MNM_OP_GRAD("mnm.op.batch_matmul", BatchMatmulGrad);

Array<Expr> DenseGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                      const Expr& dy) {
  using namespace mnm::value;
  static auto op_dense = Op::Get("mnm.op.dense");
  static auto op_transpose = Op::Get("mnm.op.transpose");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];
  const Expr& axes =
      MakeConstant(TupleValue::make(Array<Value>{IntValue::make(1), IntValue::make(0)}));
  const Expr& at = Call(op_transpose, {a, axes});
  const Expr& bt = Call(op_transpose, {b, axes});
  const Expr& dyt = Call(op_transpose, {dy, axes});
  return {Call(op_dense, {dy, bt}), Call(op_dense, {dyt, at})};
}

MNM_OP_GRAD("mnm.op.dense", DenseGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
