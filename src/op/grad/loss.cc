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

Array<Expr> NllLossGrad(const Expr& orig_call, const Expr &y, const Expr& ograds) {
  static auto dtrue = Op::Get("mnm.op.nll_loss_dtrue");
  static auto dpred = Op::Get("mnm.op.nll_loss_dpred");
  // TODO(@were): I am not sure how is the dy here.
  // CHECK_EQ(ograds.size(), 1);
  // const Expr& dy = ograds[0];
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 2);
  const Expr& true_ = call->args[0];
  const Expr& pred = call->args[1];
  return {Call(dtrue, {true_, pred}), Call(dpred, {true_, pred})};
}

MNM_OP_GRAD("mnm.op.nll_loss", NllLossGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm