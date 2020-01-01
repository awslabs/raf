/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/grad/grad_utils.cc
 * \brief Helper functions for gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

Array<Expr> AsTupleExpr(const Expr& expr, int numel) {
  if (const auto *tuple = expr.as<TupleNode>()) {
    Array<Expr> result;
    for (const Expr &expr : tuple->fields) {
      result.push_back(expr);
    }
    CHECK_GE((int) result.size(), numel);
    return result;
  }
  Array<Expr> result;
  for (int i = 0; i < numel; ++i) {
    result.push_back(TupleGetItemNode::make(expr, i));
  }
  return result;
}

}  // namespace grad
}  // namespace op
}  // namespace mnm
