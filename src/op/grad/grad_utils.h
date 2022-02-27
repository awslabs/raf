/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/grad_utils.h
 * \brief Helper functions for gradients
 */
#pragma once
#include "raf/ir.h"
#include "raf/op.h"

namespace raf {
namespace op {
namespace grad {

Expr GetShape(const Expr& expr);

Expr GetReshapeLike(const Expr& x, const Expr& like_type);

ir::Expr GetCollapseSumLike(const ir::Expr& x, const ir::Expr& like_type);

ir::Array<ir::Expr> AsTupleExpr(const ir::Expr& expr, int numel);

template <size_t n>
ir::Array<ir::Expr> NoGrads(const ir::Expr& orig_call, const ir::Array<ir::Expr> orig_args,
                            const ir::Var& y, const ir::Expr& dy) {
  return ir::Array<ir::Expr>(n, ir::NullValue<ir::Expr>());
}

}  // namespace grad
}  // namespace op
}  // namespace raf
