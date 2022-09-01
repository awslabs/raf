/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file type_erase.cc
 * \brief The type erase pass erases the checked type and function return type.
 */

#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "raf/pass_manager.h"

namespace raf {
namespace pass {
namespace type_erase {

using namespace raf::op;
using namespace raf::value;

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
  return CreateRAFFunctionPass(pass_func, 1, "EraseType", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.EraseType").set_body_typed([]() { return EraseType(); });

}  // namespace pass
}  // namespace raf
