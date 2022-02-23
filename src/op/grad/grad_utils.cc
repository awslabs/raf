/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/grad_utils.cc
 * \brief Helper functions for gradients
 */
#include "./grad_utils.h"

namespace mnm {
namespace op {
namespace grad {

using namespace mnm::ir;

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
}  // namespace mnm
