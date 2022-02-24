/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/algorithm.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "raf/pass.h"
#include "raf/ir.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

/*
To elabrate the computation of sort grad:
  x = [1,4,2,3,5]
  y = [1,2,3,4,5]
the mapping state is:
  x[0] <=> y[0]
  x[1] <=> y[3]
  x[2] <=> y[1]
  x[3] <=> y[2]
  x[4] <=> y[4]
Therefore we want the array [0,3,1,2,4] so that we can do gather to gain
the grad.
To get this array, we first do one argsort
x2y = argsort(data, axis) #[0,2,3,1,4]
this reveal the mapping:
  x[0] <=> y[0]
  x[2] <=> y[1]
  x[3] <=> y[2]
  x[1] <=> y[3]
  x[4] <=> y[4]
To get the right sequence of x, we can do another argsort:
y2x = argsort(x2y,axis) #[0,3,1,2,4]
  x[0] <=> y[0]
  x[1] <=> y[3]
  x[2] <=> y[1]
  x[3] <=> y[2]
  x[4] <=> y[4]
Then we can adopt gather op to get the results.
*/
Array<Expr> SortGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_argsort = Op::Get("raf.op.argsort");
  static auto op_gather = Op::Get("raf.op.gather");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& is_ascend = call->args[2];

  Expr x2y = Call(op_argsort, {data, axis, is_ascend});
  Expr y2x = Call(op_argsort, {x2y, axis, is_ascend});
  return {Call(op_gather, {dy, axis, y2x})};
}

RAF_OP_GRAD("raf.op.sort", SortGrad);

Array<Expr> TopkGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                     const Expr& dy) {
  static auto op_topk = Op::Get("raf.op.topk");
  static auto op_gather_dx = Op::Get("raf.op.gather_dx");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& k = call->args[1];
  const Expr& axis = call->args[2];
  const Expr& ret_type = orig_args[3];
  const Expr& is_ascend = call->args[4];
  const Expr& dtype = call->args[5];

  const auto* ret_type_const = ret_type.as<ConstantNode>();
  if (ret_type_const) {
    const auto* ret_type_str = ret_type_const->value.as<value::StringValueObj>();
    CHECK(ret_type_str && ret_type_str->value == "both")
        << "TopKGrad only supports \"both\" as return type. ret_type = " << ret_type_str->value;
  }

  // There are two options to get data indices here.
  // 1) Reuse the results in the forwarding (more memory)
  // 2) Recompute the algorithm (more latency)
  // The current implementation is option 2
  Expr data_indices = Call(op_topk, {data, k, axis, ret_type, is_ascend, dtype});
  Expr indices, dy_tensor, result;
  indices = TupleGetItem(data_indices, 1);
  dy_tensor = TupleGetItem(dy, 0);
  result = Call(op_gather_dx, {data, axis, indices, dy_tensor});

  return {result};
}

RAF_OP_GRAD("raf.op.topk", TopkGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
