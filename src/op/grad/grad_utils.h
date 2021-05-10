/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/grad_utils.h
 * \brief Helper functions for gradients
 */
#pragma once
#include "mnm/ir.h"
#include "mnm/op.h"

namespace mnm {
namespace op {
namespace grad {

ir::Array<ir::Expr> AsTupleExpr(const ir::Expr& expr, int numel);

template <size_t n>
ir::Array<ir::Expr> NoGrads(const ir::Expr& orig_call, const ir::Array<ir::Expr> orig_args,
                            const ir::Var& y, const ir::Expr& dy) {
  return ir::Array<ir::Expr>(n, ir::NullValue<ir::Expr>());
}

}  // namespace grad
}  // namespace op
}  // namespace mnm
