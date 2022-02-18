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
 * \file to_dataflow.cc
 * \brief Convert A-normal form to dataflow graph.
 */
#include <unordered_map>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "mnm/pass.h"
#include "support/arena.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op_attr_types.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace to_graph_normal_form {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm::support;

class UseVarVisitor : public ExprVisitor {
 public:
  explicit UseVarVisitor(const Var& v) : v_(v) {
  }

  static bool UseVar(const Var& v, const Expr& e) {
    UseVarVisitor uv(v);
    uv(e);
    return uv.use_var_;
  }

 private:
  bool use_var_ = false;
  Var v_;

  void VisitExpr_(const VarNode* vn) override {
    use_var_ = use_var_ || (v_ == GetRef<Var>(vn));
  }
};

class GNFConverter : public MixedModeMutator {
 public:
  Expr VisitExpr_(const LetNode* ln) final {
    Expr body = GetRef<Let>(ln);
    std::vector<std::pair<Var, Expr>> scopes;

    // Iteratively visit let nodes to avoid stack overflow.
    while (body->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(body);
      auto var = let->var;
      auto new_value = VisitExpr(let->value);

      // Check whether the let-binding var has shared memory (similar to ref-related ndoes).
      bool has_may_share = false;
      if (auto extended_var = var.as<ExtendedVarNode>()) {
        has_may_share = extended_var->may_share.defined();
      }

      if (has_may_share || new_value->IsInstance<RefCreateNode>() ||
          new_value->IsInstance<RefReadNode>() || new_value->IsInstance<RefWriteNode>()) {
        // Keep the Let for ref-related nodes as the order affects the correctness.
        scopes.emplace_back(var, new_value);
        body = let->body;
      } else {
        new_value = WrapRec(var, new_value);
        let_map_.emplace(var.get(), new_value);
        scopes.emplace_back(Var(), Expr());
        body = let->body;
      }
    }
    Expr new_body = VisitExpr(body);
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      if (it->first.defined()) {
        new_body = Let(it->first, it->second, new_body);
      }
    }
    return new_body;
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = let_map_.find(var);
    if (it != let_map_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

 private:
  Expr WrapRec(const Var& var, const Expr& val) {
    bool use_var = UseVarVisitor::UseVar(var, val);
    return use_var ? Let(var, val, var) : val;
  }

  std::unordered_map<const VarNode*, Expr> let_map_;
};
}  // namespace to_graph_normal_form

Pass ToGraphNormalForm() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(to_graph_normal_form::GNFConverter().Mutate(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "ToGraphNormalForm", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.ToGraphNormalForm").set_body_typed(ToGraphNormalForm);

}  // namespace pass
}  // namespace mnm
