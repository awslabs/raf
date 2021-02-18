/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/algorithm.cc
 * \brief Declaration of gradients
 */
#include "./grad_utils.h"
#include "mnm/pass.h"
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

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
  static auto op_argsort = Op::Get("mnm.op.argsort");
  static auto op_gather = Op::Get("mnm.op.gather");
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& data = call->args[0];
  const Expr& axis = call->args[1];
  const Expr& is_ascend = call->args[2];

  Expr x2y = Call(op_argsort, {data, axis, is_ascend});
  Expr y2x = Call(op_argsort, {x2y, axis, is_ascend});
  return {Call(op_gather, {dy, axis, y2x})};
}

MNM_OP_GRAD("mnm.op.sort", SortGrad);

}  // namespace grad
}  // namespace op
}  // namespace mnm
