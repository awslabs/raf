/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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

Array<Expr> SmoothL1LossGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                             const Expr& ograds) {
  static auto dtrue = Op::Get("mnm.op.smooth_l1_loss_dtrue");
  static auto dpred = Op::Get("mnm.op.smooth_l1_loss_dpred");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {Call(dpred, {true_, pred}), Call(dtrue, {true_, pred})};
}

MNM_OP_GRAD("mnm.op.smooth_l1_loss", SmoothL1LossGrad);

Array<Expr> NllLossGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                        const Expr& ograds) {
  static auto dpred = Op::Get("mnm.op.nll_loss_dpred");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {NullValue<Expr>(), Call(dpred, {ograds, true_, pred})};
}

MNM_OP_GRAD("mnm.op.nll_loss", NllLossGrad);

Array<Expr> CrossEntropyGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                             const Expr& ograds) {
  static auto dtrue = Op::Get("mnm.op.cross_entropy_dtrue");
  static auto dpred = Op::Get("mnm.op.cross_entropy_dpred");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {Call(dtrue, {true_, pred}), Call(dpred, {true_, pred})};
}

MNM_OP_GRAD("mnm.op.cross_entropy", CrossEntropyGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
