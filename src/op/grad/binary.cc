/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;
using namespace mnm::value;

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

  auto f = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dy, x});
    Call keep = Call(collapse_keep, {dy, x});
    return Call(sum, {dy, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  return {f(x1), f(x2)};
}

MNM_OP_GRAD("mnm.op.add", AddGrad);

Array<Expr> SubGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];

  auto f = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dy, x});
    Call keep = Call(collapse_keep, {dy, x});
    return Call(sum, {dy, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  auto fs = [&dy](const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    static auto neg = Op::Get("mnm.op.negative");
    Call axes = Call(collapse_axis, {dy, x});
    Call keep = Call(collapse_keep, {dy, x});
    Call value = Call(sum, {dy, axes, keep, MakeConstant(BoolValue::make(false))});
    return Call(neg, {value});
  };
  return {f(x1), fs(x2)};
}
MNM_OP_GRAD("mnm.op.subtract", SubGrad);

Array<Expr> RightshiftGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  // give zero gradient for any input gradient
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("mnm.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}
MNM_OP_GRAD("mnm.op.right_shift", RightshiftGrad);

Array<Expr> LeftShiftGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("mnm.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}

MNM_OP_GRAD("mnm.op.left_shift", LeftShiftGrad);

Array<Expr> MulGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_multiply = Op::Get("mnm.op.multiply");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];

  auto f = [](const Expr& dx, const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dx, x});
    Call keep = Call(collapse_keep, {dx, x});
    return Call(sum, {dx, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  return {f(Call(op_multiply, {dy, x2}), x1), f(Call(op_multiply, {dy, x1}), x2)};
}

MNM_OP_GRAD("mnm.op.multiply", MulGrad);

Array<Expr> PowGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_power = Op::Get("mnm.op.power");
  static auto op_multiply = Op::Get("mnm.op.multiply");
  static auto op_log = Op::Get("mnm.op.log");
  static auto op_divide = Op::Get("mnm.op.divide");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  Call y1 = Call(op_power, {x1, x2});
  Call y2 = Call(op_divide, {y1, x1});
  Call dx1 = Call(op_multiply, {x2, y2});
  Call x1_log = Call(op_log, {x1});
  Call dx2 = Call(op_multiply, {y1, x1_log});

  auto f = [](const Expr& dx, const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dx, x});
    Call keep = Call(collapse_keep, {dx, x});
    return Call(sum, {dx, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  return {f(Call(op_multiply, {dy, dx1}), x1), f(Call(op_multiply, {dy, dx2}), x2)};
}

MNM_OP_GRAD("mnm.op.power", PowGrad);

Array<Expr> DivGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                    const Expr& dy) {
  static auto op_divide = Op::Get("mnm.op.divide");
  static auto op_multiply = Op::Get("mnm.op.multiply");
  static auto op_negative = Op::Get("mnm.op.negative");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  Call dx1 = Call(op_divide, {dy, x2});
  Call dx2 = Call(op_negative, {dy});
  dx2 = Call(op_multiply, {dx2, Call(op_divide, {x1, x2})});
  dx2 = Call(op_divide, {dx2, x2});

  auto f = [](const Expr& dx, const Expr& x) {
    static auto collapse_axis = Op::Get("mnm.op.get_reduce_axis");
    static auto collapse_keep = Op::Get("mnm.op.get_kept_dims");
    static auto sum = Op::Get("mnm.op.sum");
    Call axes = Call(collapse_axis, {dx, x});
    Call keep = Call(collapse_keep, {dx, x});
    return Call(sum, {dx, axes, keep, MakeConstant(BoolValue::make(false))});
  };

  return {f(dx1, x1), f(dx2, x2)};
}

MNM_OP_GRAD("mnm.op.divide", DivGrad);

Array<Expr> FloorDivGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                         const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto f = [&dy](const Expr& x) {
    static auto op_zeros_like = Op::Get("mnm.op.zeros_like");
    Call zero = Call(op_zeros_like, {x});
    return zero;
  };

  return {f(x1), f(x2)};
}

MNM_OP_GRAD("mnm.op.floor_divide", FloorDivGrad);

Array<Expr> BinaryZeroGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                           const Expr& dy) {
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& x1 = call->args[0];
  const Expr& x2 = call->args[1];
  auto MakeZero = [](const Expr expr) {
    static auto zeros_like = Op::Get("mnm.op.zeros_like");
    auto zero_grad = Call(zeros_like, {expr});
    return zero_grad;
  };
  return {MakeZero(x1), MakeZero(x2)};
}

MNM_OP_GRAD("mnm.op.not_equal", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.equal", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.less", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.less_equal", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.greater", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.greater_equal", BinaryZeroGrad);
MNM_OP_GRAD("mnm.op.logical_and", BinaryZeroGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
