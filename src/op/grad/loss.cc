/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/binary.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Array<Expr> SmoothL1LossGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                             const Expr& ograds) {
  static auto dtrue = Op::Get("raf.op.smooth_l1_loss_dtrue");
  static auto dpred = Op::Get("raf.op.smooth_l1_loss_dpred");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {Call(dpred, {true_, pred}), Call(dtrue, {true_, pred})};
}

RAF_OP_GRAD("raf.op.smooth_l1_loss", SmoothL1LossGrad);

Array<Expr> NllLossGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                        const Expr& ograds) {
  static auto dpred = Op::Get("raf.op.nll_loss_dpred");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {NullValue<Expr>(), Call(dpred, {ograds, true_, pred})};
}

RAF_OP_GRAD("raf.op.nll_loss", NllLossGrad);

Array<Expr> CrossEntropyGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                             const Expr& ograds) {
  using namespace raf::value;
  static auto op_exp = Op::Get("raf.op.exp");
  static auto op_sum = Op::Get("raf.op.sum");
  static auto op_multiply = Op::Get("raf.op.multiply");
  static auto op_subtract = Op::Get("raf.op.subtract");
  static auto dpred = Op::Get("raf.op.nll_loss_dpred");
  static auto log_softmax = Op::Get("raf.op.log_softmax");

  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  Expr keep_dims = MakeConstant(ScalarValue::make((int64_t)1));

  auto nll_loss_dpred = Call(dpred, {ograds, true_, pred});
  auto log_pred = Call(log_softmax, {pred});

  Expr e_1 = Call(op_sum, {nll_loss_dpred, MakeConstant(ScalarValue::make((int64_t)1)), keep_dims,
                           MakeConstant(BoolValue::make(false))});
  Expr e_2 = Call(op_multiply, {Call(op_exp, {log_pred}), e_1});
  Expr e_3 = Call(op_subtract, {nll_loss_dpred, e_2, MakeNull(), MakeNull()});
  return {NullValue<Expr>(), e_3};
}

RAF_OP_GRAD("raf.op.cross_entropy", CrossEntropyGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
