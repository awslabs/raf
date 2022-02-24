/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file inline_backward.cc
 * \brief inlining backward graph in the forward pass
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace inline_backward {

using namespace raf::ir;

class InlineBackwardFunc : public ExprVisitor {
 public:
  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* let) {
      if (!in_closure_ && let->body.get() == let->var.get()) {
        // The return stmt in original function
        if (auto tup = let->value.as<TupleNode>()) {
          old_ret_ = tup->fields;
        }
      } else if (let->value.as<FunctionNode>()) {
        // The backward graph closure
        closure_var_ = let->var.get();
      } else {
        ell_.exprs.push_back(let->value);
        ell_.vars.push_back(let->var);
      }
      this->VisitExpr(let->var);
      this->VisitExpr(let->value);
    };
    auto post_visit = [this](const LetNode* let) {
      this->VisitExpr(let->body);
      this->visit_counter_[let] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const FunctionNode* func_node) final {
    closure_params_ = func_node->params;
    in_closure_ = true;
    VisitExpr(func_node->body);
    in_closure_ = false;
  }

  Function Inline(Function func) {
    VisitExpr(func->body);
    if (closure_var_ == nullptr) {
      // No closure found in the function
      return func;
    }
    // Check if the closure is in the return tuple
    bool return_closure = false;
    Array<Expr> ret_tup = old_ret_;
    for (auto it = ret_tup.begin(); it != ret_tup.end(); ++it) {
      if ((*it).get() == closure_var_) {
        ret_tup.erase(it);
        return_closure = true;
        break;
      }
    }
    if (!return_closure) {
      return func;
    }
    // Append the closure params into the function params
    auto new_params = func->params;
    for (auto closure_param : closure_params_) {
      new_params.push_back(closure_param);
    }
    // Rename the closure ret var to be gradient
    Var gradient = MakeVar("gradient", {});
    ell_.vars[ell_.vars.size() - 1] = gradient;
    ret_tup.push_back(ell_.vars.back());
    // Add an extra return stmt
    Var ret_var = MakeVar("ret", {});
    ell_.vars.push_back(ret_var);
    ell_.exprs.push_back(Tuple(ret_tup));
    ell_.ret = ret_var;
    return Function(new_params, ell_.AsExpr(), {}, {});
  }

 private:
  /*! \brief Original return value for function */
  Array<Expr> old_ret_;
  /*! \brief Closure parameters */
  Array<Var> closure_params_;
  /*! \brief Pointers to store the var for closure */
  const VarNode* closure_var_ = nullptr;
  /*! \brief ExplicitLetList to rebuild the function */
  ExplicitLetList ell_;
  /*! \brief Indicate whether it is in closure */
  bool in_closure_ = false;
};
}  // namespace inline_backward

Pass InlineBackward() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return inline_backward::InlineBackwardFunc().Inline(f);
  };
  return CreateRAFFunctionPass(pass_func, 1, "InlineBackward", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.InlineBackward").set_body_typed(InlineBackward);

}  // namespace pass
}  // namespace raf
