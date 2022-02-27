/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/reduce.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Array<Expr> MeanGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                     const Expr& dy) {
  static auto mean_dx = Op::Get("raf.op.mean_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  return {Call(mean_dx, {dy, GetShape(call->args[0]), axis, keepdims, exclude})};
}

RAF_OP_GRAD("raf.op.mean", MeanGrad);

Array<Expr> SumGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                    const Expr& dy) {
  static auto sum_dx = Op::Get("raf.op.sum_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  return {Call(sum_dx, {x, dy, axis, keepdims, exclude})};
}

RAF_OP_GRAD("raf.op.sum", SumGrad);

Array<Expr> ProdGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                     const Expr& dy) {
  static auto prod_dx = Op::Get("raf.op.prod_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  return {Call(prod_dx, {x, dy, axis, keepdims, exclude})};
}

RAF_OP_GRAD("raf.op.prod", ProdGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
