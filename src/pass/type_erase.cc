/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
