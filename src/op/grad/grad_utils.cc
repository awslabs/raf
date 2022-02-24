/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/grad_utils.cc
 * \brief Helper functions for gradients
 */
#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

Expr GetShape(const Expr& expr) {
  static auto op_shape = Op::Get("raf.op.shape");
  static auto op_size = Op::Get("raf.op.shape_as_tensor");
  if (expr->checked_type_.defined() && tvm::relay::IsDynamic(expr->checked_type_)) {
    return Call(op_size, {expr});
  }
  return Call(op_shape, {expr});
}

Expr GetReshapeLike(const Expr& x, const Expr& like_type) {
  static auto reshape_like = Op::Get("raf.op.reshape_like");
  static auto reshape = Op::Get("raf.op.reshape");
  static auto shape = Op::Get("raf.op.shape");
  if (like_type->checked_type_.defined() && tvm::relay::IsDynamic(like_type->checked_type_)) {
    return {Call(reshape_like, {x, like_type})};
  }
  return {Call(reshape, {x, Call(shape, {like_type})})};
}

Expr GetCollapseSumLike(const Expr& x, const Expr& like_type) {
  static auto collapse_axis = Op::Get("raf.op.get_reduce_axis");
  static auto collapse_keep = Op::Get("raf.op.get_kept_dims");
  static auto sum = Op::Get("raf.op.sum");
  static auto collapse_sum_like = Op::Get("raf.op.collapse_sum_like");
  if (like_type->checked_type_.defined() && tvm::relay::IsDynamic(like_type->checked_type_)) {
    return Call(collapse_sum_like, {x, like_type});
  }
  Call axes = Call(collapse_axis, {x, like_type});
  Call keep = Call(collapse_keep, {x, like_type});
  return Call(sum, {x, axes, keep, MakeConstant(value::BoolValue::make(false))});
};

Array<Expr> AsTupleExpr(const Expr& expr, int numel) {
  if (const auto* tuple = expr.as<TupleNode>()) {
    Array<Expr> result;
    for (const Expr& expr : tuple->fields) {
      result.push_back(expr);
    }
    return result;
  }
  Array<Expr> result;
  for (int i = 0; i < numel; ++i) {
    result.push_back(TupleGetItem(expr, i));
  }
  return result;
}

}  // namespace grad
}  // namespace op
}  // namespace raf
