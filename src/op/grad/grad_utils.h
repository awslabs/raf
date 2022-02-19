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
