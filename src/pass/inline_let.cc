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
 * \file src/pass/inline_let.cc
 * \brief Inline the Let stmt when a variable is assigned to another variable or a TupleGetItem
 * expression that can be simplified.
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"

namespace mnm {
namespace pass {
namespace inline_let {

using namespace mnm::ir;
using namespace mnm::op;

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
  return CreateMNMFunctionPass(pass_func, 1, "InlineLet", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.InlineLet").set_body_typed(InlineLet);

}  // namespace pass
}  // namespace mnm
