/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/pass/to_a_normal_form.cc
 * \brief Convert dataflow graph to A-normal form.
 */
#include <vector>
#include <unordered_map>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./convert_utils.h"
namespace raf {
namespace pass {

using raf::analysis::CreateDependencyGraph;
using tvm::relay::CalcScope;

Expr Fill::ToANormalForm(const Expr& e, const DependencyGraph& dg, NodeScopeMap* node_scope) {
  Fill fi(dg, node_scope, nullptr);
  return fi.GetScope(e)->let_list->Get(fi.VisitExpr(e));
}

Scope Fill::GetScope(const Expr& e) {
  return node_scope_->at(dg_.expr_node.at(e));
}

Scope Fill::GetSubScope(const Expr& e, size_t i) {
  DependencyGraph::Node* n = dg_.expr_node.at(e);
  auto h = n->children.head;
  while (i != 0) {
    ICHECK(h);
    --i;
    h = h->next;
  }
  ICHECK(h);
  return node_scope_->at(h->value);
}

Expr Fill::VisitExpr(const Expr& e, const Var& v) {
  if (memo.count(e) == 0) {
    memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
  } else if (v.defined()) {
    GetScope(e)->let_list->Push(v, memo.at(e));
  }
  auto ret = memo.at(e);
  // if no include_set is specified, every expression should be atomic.
  if (include_set_ == nullptr) ICHECK(IsAtomic(ret)) << ret->GetTypeKey();
  return ret;
}

Expr Fill::VisitExpr(const Expr& e) {
  return this->VisitExpr(e, Var());
}

Expr Fill::Atomic(const Expr& e, const Var& v) {
  return v.defined() ? GetScope(e)->let_list->Push(v, e) : e;
}

// Bind expression `now` to var `v` if the original expression is in the include set, or if
// v is already defined (e.g. coming from a Let expression). Otherwise return `now` directly.
Expr Fill::Compound(const Expr& orig, const Expr& now, const Var& v) {
  Var var = v.defined() ? v : MakeVar("x" + std::to_string(++num_created_var_), Type());
  bool not_included = include_set_ && include_set_->find(orig) == include_set_->end();
  if (!v.defined() && not_included) {
    return now;
  } else {
    auto let_list = GetScope(orig)->let_list;
    if (auto let = now.as<LetNode>()) {
      // If the expression is a Let, then we inline the expression LetList directly.
      Expr ret, body;
      do {
        ret = let_list->Push(let->var, let->value);
        body = let->body;
        let = body.as<LetNode>();
      } while (let);
      return ret;
    }
    return let_list->Push(var, now);
  }
}

Expr Fill::VisitExpr_(const CallNode* c, const Var& v) {
  Expr e = GetRef<Expr>(c);
  std::vector<Expr> args;
  for (const auto& a : c->args) {
    args.push_back(VisitExpr(a));
  }
  return Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
}

Expr Fill::VisitExpr_(const TupleNode* t, const Var& v) {
  Expr e = GetRef<Expr>(t);
  std::vector<Expr> fields;
  for (const auto& a : t->fields) {
    fields.push_back(VisitExpr(a));
  }
  return Compound(e, Tuple(fields), v);
}

Expr Fill::VisitExpr_(const TupleGetItemNode* t, const Var& v) {
  Expr e = GetRef<Expr>(t);
  return Compound(e, TupleGetItem(VisitExpr(t->tuple), t->index), v);
}

Expr Fill::VisitExpr_(const RefCreateNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefCreate(VisitExpr(r->value)), v);
}

Expr Fill::VisitExpr_(const RefReadNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefRead(VisitExpr(r->ref)), v);
}

Expr Fill::VisitExpr_(const RefWriteNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefWrite(VisitExpr(r->ref), VisitExpr(r->value)), v);
}

Expr Fill::VisitExpr_(const IfNode* i, const Var& v) {
  Expr e = GetRef<Expr>(i);
  Expr ret = If(VisitExpr(i->cond), GetSubScope(e, 1)->let_list->Get(VisitExpr(i->true_branch)),
                GetSubScope(e, 2)->let_list->Get(VisitExpr(i->false_branch)));
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const FunctionNode* f, const Var& v) {
  Expr e = GetRef<Expr>(f);
  Expr ret;
  if (f->HasNonzeroAttr(attr::kPrimitive)) {
    ret = e;
  } else {
    ret = Function(f->params, GetSubScope(e, 0)->let_list->Get(VisitExpr(f->body)), f->ret_type,
                   f->type_params, f->attrs);
  }
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const LetNode* l, const Var& v) {
  Expr e = GetRef<Expr>(l);
  VisitExpr(l->value, l->var);
  Expr ret = GetSubScope(e, 0)->let_list->Get(VisitExpr(l->body));
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const RelayConstantNode* c, const Var& v) {
  Expr e = GetRef<Expr>(c);
  return Atomic(e, v);
}

Expr Fill::VisitExpr_(const VarNode* vn, const Var& v) {
  Expr e = GetRef<Expr>(vn);
  return Atomic(e, v);
}

Expr Fill::VisitExpr_(const GlobalVarNode* gvn, const Var& v) {
  GlobalVar gv = GetRef<GlobalVar>(gvn);
  return Atomic(gv, v);
}

Expr Fill::VisitExpr_(const OpNode* op, const Var& v) {
  Expr e = GetRef<Expr>(op);
  return Atomic(e, v);
}

Expr ToANormalFormExpr(const Expr& expr) {
  tvm::support::Arena arena;
  DependencyGraph dg = CreateDependencyGraph(&arena, expr, false);
  /* In order to model new subscopes created by lambda, if else and pattern matching,
   * we also assign scope to edge as well.
   * The scope of an edge is either the parent's scope, or a new subscope of the parent's scope.
   *
   * So, the scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   *
   * Every scope additionally contain a LetList which collect all value of that scope.
   * We do an additional pass to fill all the LetList and we are done.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToANormalForm(expr, dg, &scopes.first);
}

IRModule ToANormalFormAux(IRModule m) {
  tvm::Map<GlobalVar, BaseFunc> updates;
  auto funcs = m->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToANormalFormExpr(e); },
                          Downcast<Function>(it.second));
    ICHECK_EQ(FreeVars(ret).size(), 0)
        << ir::AsText(ret) << "should not has free vars: " << FreeVars(ret);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    m->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToANF: transformed" << std::endl << m;

  return m;
}

Pass ToANormalForm() {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    return ToANormalFormAux(m);
  };
  return CreateModulePass(pass_func, 1, "ToANormalForm", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ToANormalForm").set_body_typed(ToANormalForm);

}  // namespace pass
}  // namespace raf
