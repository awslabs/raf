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
 * \file type_erase.cc
 * \brief The type erase pass erases the checked type and function return type.
 */

#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/pass_manager.h"

namespace mnm {
namespace pass {
namespace type_erase {

using namespace mnm::op;
using namespace mnm::value;

class TypeEraser : public ExprMutator {
 public:
  Expr VisitExpr(const Expr& expr) final {
    auto ret = ExprMutator::VisitExpr(expr);
    ret->checked_type_ = Type();
    return ret;
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    return Function(node->params, node->body, Type(), node->type_params, node->attrs, node->span);
  }
};

}  // namespace type_erase

Pass EraseType() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(type_erase::TypeEraser().Mutate(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "EraseType", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.EraseType").set_body_typed([]() { return EraseType(); });

}  // namespace pass
}  // namespace mnm
