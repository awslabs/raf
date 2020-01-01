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
}  // namespace grad
}  // namespace op
}  // namespace mnm
