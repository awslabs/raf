/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file extract_binding.cc
 * \brief Extracting a relay body from frontend defined binding
 */
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/binding.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace extract_binding {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::binding;

class Extractor final : public ExprVisitor {
 public:
  explicit Extractor(const Array<Var>& ignores) {
    for (const Var& var : ignores) {
      this->ignore.insert(var.get());
    }
  }

  void VisitExpr_(const VarNode* var) final {
    LOG(FATAL) << "Should not be here";
  }

  void VisitExpr_(const CallNode* node) final {
    EnqueueVar(node->op);
    for (const Expr& expr : node->args) {
      EnqueueVar(expr);
    }
  }

  void VisitExpr_(const TupleNode* node) final {
    for (const Expr& expr : node->fields) {
      EnqueueVar(expr);
    }
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    EnqueueVar(node->tuple);
  }

  void VisitExpr_(const IfNode* node) final {
    EnqueueVar(node->cond);
    EnqueueVar(node->true_branch);
    EnqueueVar(node->false_branch);
  }

  void VisitExpr_(const FunctionNode* node) final {
    for (const Var& var : FreeVars(GetRef<Function>(node))) {
      EnqueueVar(var);
    }
  }

  void EnqueueVar(const Expr& expr) {
    if (expr->IsInstance<ConstantNode>() || expr->IsInstance<OpNode>()) {
      return;
    }
    if (phase == 0) {
      if (const VarNode* var = expr.as<VarNode>()) {
        if (++in_degree[var] == 1) {
          queue.push_back(var);
        }
      } else {
        LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
      }
      return;
    }
    if (phase == 1) {
      if (const VarNode* var = expr.as<VarNode>()) {
        if (--in_degree[var] == 0) {
          queue.push_back(var);
        }
      } else {
        LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
      }
      return;
    }
    LOG(FATAL) << "Shouldn't be here";
    throw;
  }

  std::vector<const VarNode*> queue;
  std::unordered_map<const VarNode*, int> in_degree;
  std::unordered_map<const VarNode*, const ExprNode*> bindings;
  std::unordered_set<const VarNode*> ignore;
  int phase{};

  Expr Run(const Var& var) {
    // Calculate the in_degree of each var
    // Basically in_degree means how many times the var is used in other exprs
    phase = 0;
    EnqueueVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      queue.pop_back();
      if (ignore.find(var) != ignore.end()) {
        continue;
      }
      const auto& binding = LookupBinding(var);
      CHECK(binding.defined()) << "Unbinded variable " << GetRef<Var>(var);
      if (const auto* sym = binding.as<SymbolBindingObj>()) {
        const Expr& expr = sym->expr;
        bindings[var] = expr.operator->();
        if (expr.defined()) {
          ExprVisitor::VisitExpr(expr);
        }
      } else if (binding->IsInstance<NDArrayBindingObj>()) {
        bindings[var] = {};
        continue;
      }
    }
    // Do topo-sort based on in_degree
    // If the in_degree of a var decreases to 0,
    // it means the var can be bound without being malformed
    phase = 1;
    Expr body = var;
    queue.clear();
    this->visit_counter_.clear();
    EnqueueVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const ExprNode* expr_node = bindings[var];
      queue.pop_back();
      if (expr_node == nullptr) {
        continue;
      }
      if (!expr_node->IsInstance<ConstantNode>()) {
        body = Let(GetRef<Var>(var), GetRef<Expr>(expr_node), body);
      }
      ExprVisitor::VisitExpr(GetRef<Expr>(expr_node));
    }
    return body;
  }
};

Expr ExtractBinding(const Var& var, const Array<Var>& ignore) {
  return Extractor(ignore).Run(var);
}

RAF_REGISTER_GLOBAL("raf.pass_.ExtractBinding").set_body_typed(ExtractBinding);
}  // namespace extract_binding
}  // namespace pass
}  // namespace raf
