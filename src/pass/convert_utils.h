/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file convert_utils.h
 * \brief Utilities for converting to A-normal form and basic block normal form.
 */
#pragma once

#include "relay/analysis/dependency_graph.h"
#include "relay/transforms/pass_utils.h"
#include "tvm/relay/transform.h"

namespace raf {
namespace pass {

using tvm::relay::DependencyGraph;
using tvm::relay::ExprSet;
using tvm::relay::IsAtomic;
using tvm::relay::NodeScopeMap;
using tvm::relay::Scope;

using namespace raf::ir;

// TODO(@icemelon9): Inherit from Relay Fill after changing it in the upstream
class Fill : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToANormalForm(const Expr& e, const DependencyGraph& dg, NodeScopeMap* node_scope);

  // For basic block normal form, bind expressions only if the original expression's
  // scope should be lifted
  static Expr ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                                     NodeScopeMap* node_scope, ExprSet* lifted);

 private:
  const DependencyGraph& dg_;
  NodeScopeMap* node_scope_ = nullptr;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo;
  // a set of Expressions to include for let bindings. If set to nullptr
  // all Exprs will be pushed to the let list.
  ExprSet* include_set_ = nullptr;

  Fill(const DependencyGraph& dg, NodeScopeMap* node_scope, ExprSet* include_set)
      : dg_(dg), node_scope_(node_scope), include_set_(include_set) {
  }

  Scope GetScope(const Expr& e);
  Scope GetSubScope(const Expr& e, size_t i);

  Expr VisitExpr(const Expr& e, const Var& v) final;
  Expr VisitExpr(const Expr& e);

  Expr Atomic(const Expr& e, const Var& v);
  // Bind expression `now` to var `v` if the original expression is in the include set, or if
  // v is already defined (e.g. coming from a Let expression). Otherwise return `now` directly.
  Expr Compound(const Expr& orig, const Expr& now, const Var& v);

  Expr VisitExpr_(const CallNode* c, const Var& v) final;
  Expr VisitExpr_(const TupleNode* t, const Var& v) final;
  Expr VisitExpr_(const TupleGetItemNode* t, const Var& v) final;
  Expr VisitExpr_(const RefCreateNode* r, const Var& v) final;
  Expr VisitExpr_(const RefReadNode* r, const Var& v) final;
  Expr VisitExpr_(const RefWriteNode* r, const Var& v) final;
  Expr VisitExpr_(const IfNode* i, const Var& v) final;
  Expr VisitExpr_(const FunctionNode* f, const Var& v) final;
  Expr VisitExpr_(const LetNode* l, const Var& v) final;
  Expr VisitExpr_(const RelayConstantNode* c, const Var& v) final;
  Expr VisitExpr_(const VarNode* vn, const Var& v) final;
  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final;
  Expr VisitExpr_(const OpNode* op, const Var& v) final;

  /*! \brief The number of created vars used to generate unique name hints. */
  size_t num_created_var_ = 0;
};

}  // namespace pass
}  // namespace raf
