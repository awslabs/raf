/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "raf/op_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;
using namespace raf::value;

Array<Expr> AddGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  // schema for binary is:
  //    x1, x2
  // schema for binary_dx is:
  //    x1, x2, y, dy
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  return {GetCollapseSumLike(dy, x1), GetCollapseSumLike(dy, x2)};
}

RAF_OP_GRAD("raf.op.add", AddGrad);

Array<Expr> SubGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];

  auto fs = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("raf.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("raf.op.get_kept_dims");
    static auto sum = Op::Get("raf.op.sum");
    static auto neg = Op::Get("raf.op.negative");
    Call axes = Call(collapse_axis, {dy, x});
    Call keep = Call(collapse_keep, {dy, x});
    Call value = Call(sum, {dy, axes, keep, MakeConstant(BoolValue::make(false))});
    return Call(neg, {value});
  };
  return {GetCollapseSumLike(dy, x1), fs(x2)};
}
RAF_OP_GRAD("raf.op.subtract", SubGrad);

Array<Expr> RightshiftGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  // give zero gradient for any input gradient
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("raf.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}
RAF_OP_GRAD("raf.op.right_shift", RightshiftGrad);

Array<Expr> LeftShiftGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("raf.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}

RAF_OP_GRAD("raf.op.left_shift", LeftShiftGrad);

Array<Expr> MulGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_multiply = Op::Get("raf.op.multiply");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  return {GetCollapseSumLike(Call(op_multiply, {dy, x2}), x1),
          GetCollapseSumLike(Call(op_multiply, {dy, x1}), x2)};
}

RAF_OP_GRAD("raf.op.multiply", MulGrad);

Array<Expr> PowGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_power = Op::Get("raf.op.power");
  static auto op_ones_like = Op::Get("raf.op.ones_like");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_log = Op::Get("raf.op.log");
  static auto op_subtract = Op::Get("raf.op.subtract");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x = call->args[0];
  const Expr& a = call->args[1];

  // dx = a * x^(a-1)
  Call ones = Call(op_ones_like, {a});
  Call a_minus_one = Call(op_subtract, {a, ones, MakeNull(), MakeNull()});
  Call x_pow_minus_one = Call(op_power, {x, a_minus_one});
  Call dx = Call(op_multiply, {a, x_pow_minus_one});

  // da = x^a * log(x)
  // FIXME: PyTorch defines d(x^a)/da at x = 0 and a < 0 to be -inf. This is due to
  // d(x^a)/da -> -inf for fixed a as x -> +0. However, this requires either an if-branch
  // or special handling at the kernel level, so we disable the gradient of "a" for a now.
  // Call x_pow = Call(op_power, {x, a});
  // Call x_log = Call(op_log, {x});
  // Call da = Call(op_multiply, {x_pow, x_log});

  return {GetCollapseSumLike(Call(op_multiply, {dy, dx}), x), ir::NullValue<ir::Expr>()};
}

RAF_OP_GRAD("raf.op.power", PowGrad);

Array<Expr> DivGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_divide = Op::Get("raf.op.divide");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_negative = Op::Get("raf.op.negative");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  Call dx1 = Call(op_divide, {dy, x2});
  Call dx2 = Call(op_negative, {dy});
  dx2 = Call(op_multiply, {dx2, Call(op_divide, {x1, x2})});
  dx2 = Call(op_divide, {dx2, x2});

  return {GetCollapseSumLike(dx1, x1), GetCollapseSumLike(dx2, x2)};
}

RAF_OP_GRAD("raf.op.divide", DivGrad);

Array<Expr> FloorDivGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("raf.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}

RAF_OP_GRAD("raf.op.floor_divide", FloorDivGrad);

Array<Expr> BinaryZeroGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto MakeZero = [](const Expr expr) {
    static auto zeros_like = Op::Get("raf.op.zeros_like");
    auto zero_grad = Call(zeros_like, {expr});
    return zero_grad;
  };
  return {MakeZero(x1), MakeZero(x2)};
}

RAF_OP_GRAD("raf.op.not_equal", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.equal", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.less", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.less_equal", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.greater", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.greater_equal", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.logical_and", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.logical_or", BinaryZeroGrad);
RAF_OP_GRAD("raf.op.logical_xor", BinaryZeroGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
