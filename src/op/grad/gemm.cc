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
  if (!transpose_a) {
    if (!transpose_b) {
      return {
          Call(op_nt, {dy, b}),
          Call(op_tn, {a, dy}),
      };
    } else {
      return {
          Call(op_nn, {dy, b}),
          Call(op_tn, {dy, a}),
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
MNM_OP_GRAD("mnm.op.dense", MatmulGradNT);
MNM_OP_GRAD("mnm.op.matmul_tn", MatmulGradTN);
MNM_OP_GRAD("mnm.op.matmul_tt", MatmulGradTT);

template <bool transpose_a, bool transpose_b>
Array<Expr> BatchMatmulGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                                const Expr& dy) {
  static auto op_nn = Op::Get("mnm.op.batch_matmul");
  static auto op_nt = Op::Get("mnm.op.batch_matmul_nt");
  static auto op_tn = Op::Get("mnm.op.batch_matmul_tn");
  static auto op_tt = Op::Get("mnm.op.batch_matmul_tt");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& a = call->args[0];
  const Expr& b = call->args[1];

  auto f = [](const Expr& dx, const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dx, x});
    Call keep = Call(collapse_keep, {dx, x});
    return Call(sum, {dx, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  if (!transpose_a) {
    if (!transpose_b) {
      return {
          f(Call(op_nt, {dy, b}), a),
          f(Call(op_tn, {a, dy}), b),
      };
    } else {
      return {
          f(Call(op_nn, {dy, b}), a),
          f(Call(op_tn, {dy, a}), b),
      };
    }
  } else {
    if (!transpose_b) {
      return {
          f(Call(op_nt, {b, dy}), a),
          f(Call(op_nn, {a, dy}), b),
      };
    } else {
      return {
          f(Call(op_tt, {b, dy}), a),
          f(Call(op_tt, {dy, a}), b),
      };
    }
  }
  LOG(FATAL) << "Unreachable code";
  throw;
}

auto BatchMatmulGradNN = BatchMatmulGradImpl<false, false>;
auto BatchMatmulGradNT = BatchMatmulGradImpl<false, true>;
auto BatchMatmulGradTN = BatchMatmulGradImpl<true, false>;
auto BatchMatmulGradTT = BatchMatmulGradImpl<true, true>;

MNM_OP_GRAD("mnm.op.batch_matmul", BatchMatmulGradNN);
MNM_OP_GRAD("mnm.op.batch_matmul_nt", BatchMatmulGradNT);
MNM_OP_GRAD("mnm.op.batch_matmul_tn", BatchMatmulGradTN);
MNM_OP_GRAD("mnm.op.batch_matmul_tt", BatchMatmulGradTT);

}  // namespace grad
}  // namespace op
}  // namespace mnm
