/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/inline_let.cc
 * \brief Inline the Let stmt when a variable is assigned to another variable or a TupleGetItem
 * expression that can be simplified.
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"

namespace raf {
namespace pass {
namespace inline_let {

using namespace raf::ir;
using namespace raf::op;

class LetInliner : public ExprMutator {
 public:
  LetInliner() {
  }

  Expr VisitExpr_(const LetNode* let) {
    auto pre_visit = [this](const LetNode* let) {
      if (let->value->IsInstance<TupleNode>()) {
        tuple_map_.emplace(let->var, Downcast<Tuple>(let->value));
      }

      auto new_value = VisitExpr(let->value);
      if (new_value->IsInstance<VarNode>()) {
        auto alias_var = Downcast<Var>(new_value);
        alias_map_.emplace(let->var, alias_var);
      }
    };
    auto post_visit = [this](const LetNode* let) {
      Var var = Downcast<Var>(this->VisitExpr(let->var));
      Expr value = this->VisitExpr(let->value);
      Expr body = this->VisitExpr(let->body);
      auto expr = GetRef<Expr>(let);
      if (value->IsInstance<VarNode>()) {
        // If the new value is a var, then this let statement becomes "let var1 = var2",
        // which can be eliminated.
        this->memo_[expr] = body;
      } else if (var.same_as(let->var) && value.same_as(let->value) && body.same_as(let->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(let, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let)];
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    auto tup = VisitExpr(node->tuple);
    auto it = tuple_map_.find(tup);
    if (it != tuple_map_.end()) {
      Tuple tuple = it->second;
      return tuple->fields[node->index];
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr VisitExpr_(const VarNode* var) {
    Var ret = GetRef<Var>(var);
    auto it = alias_map_.find(ret);
    while (it != alias_map_.end()) {
      ret = it->second;
      it = alias_map_.find(ret);
    }
    return ret;
  }

 private:
  /*! \brief Mapping of a var to a tuple. */
  std::unordered_map<Expr, Tuple, ObjectPtrHash, ObjectPtrEqual> tuple_map_;
  /*! \brief Mapping from a var to another var. */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_map_;
};

}  // namespace inline_let

Pass InlineLet() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(inline_let::LetInliner().VisitExpr(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "InlineLet", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.InlineLet").set_body_typed(InlineLet);

}  // namespace pass
}  // namespace raf
