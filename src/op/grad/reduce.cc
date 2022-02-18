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
 * \file src/op/grad/reduce.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> MeanGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                     const Expr& dy) {
  static auto mean_dx = Op::Get("mnm.op.mean_dx");
  static auto shape = Op::Get("mnm.op.shape");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  const Expr& x_shape = Call(shape, {call->args[0]});
  return {Call(mean_dx, {dy, axis, x_shape, keepdims, exclude})};
}

MNM_OP_GRAD("mnm.op.mean", MeanGrad);

Array<Expr> SumGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                    const Expr& dy) {
  static auto sum_dx = Op::Get("mnm.op.sum_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  return {Call(sum_dx, {x, dy, axis, keepdims, exclude})};
}

MNM_OP_GRAD("mnm.op.sum", SumGrad);

Array<Expr> ProdGrad(const Expr& orig_call, const Array<Expr> orig_args, const Expr& y,
                     const Expr& dy) {
  static auto prod_dx = Op::Get("mnm.op.prod_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK_GE(call->args.size(), 3);
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& keepdims = call->args[2];
  const Expr& exclude = call->args[3];
  return {Call(prod_dx, {x, dy, axis, keepdims, exclude})};
}

MNM_OP_GRAD("mnm.op.prod", ProdGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
