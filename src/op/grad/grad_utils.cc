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
