/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/to_a_normal_form.cc
 * \brief Convert dataflow graph to A-normal form.
 */
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "relay/transforms/pass_utils.h"
#include "relay/analysis/dependency_graph.h"
#include "tvm/relay/transform.h"

namespace mnm {
namespace pass {

using tvm::relay::CalcScope;
using tvm::relay::DependencyGraph;
using tvm::relay::ExprSet;
using tvm::relay::IsAtomic;
using tvm::relay::NodeScopeMap;
using tvm::relay::Scope;
using namespace mnm::ir;

namespace to_a_normal_form {

class Fill : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToANormalForm(const Expr& e, const DependencyGraph& dg, NodeScopeMap* node_scope) {
    Fill fi(dg, node_scope, nullptr);
    return fi.GetScope(e)->let_list->Get(fi.VisitExpr(e));
  }

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

  Scope GetScope(const Expr& e) {
    return node_scope_->at(dg_.expr_node.at(e));
  }

  Scope GetSubScope(const Expr& e, size_t i) {
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

  Expr VisitExpr(const Expr& e, const Var& v) final {
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
    } else if (v.defined()) {
      GetScope(e)->let_list->Push(v, memo.at(e));
    }
    auto ret = memo.at(e);
    // if no include_set is specified, every expression should be atomic.
    if (include_set_ == nullptr) ICHECK(IsAtomic(ret));
    return ret;
  }

  Expr VisitExpr(const Expr& e) {
    return this->VisitExpr(e, Var());
  }

  Expr Atomic(const Expr& e, const Var& v) {
    return v.defined() ? GetScope(e)->let_list->Push(v, e) : e;
  }

  // Bind expression `now` to var `v` if the original expression is in the include set, or if
  // v is already defined (e.g. coming from a Let expression). Otherwise return `now` directly.
  Expr Compound(const Expr& orig, const Expr& now, const Var& v) {
    Var var = v.defined() ? v : MakeVar("x", Type());
    bool not_included = include_set_ && include_set_->find(orig) == include_set_->end();
    if (!v.defined() && not_included) {
      return now;
    } else {
      return GetScope(orig)->let_list->Push(var, now);
    }
  }

  Expr VisitExpr_(const CallNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
  }

  Expr VisitExpr_(const TupleNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    std::vector<Expr> fields;
    for (const auto& a : t->fields) {
      fields.push_back(VisitExpr(a));
    }
    return Compound(e, Tuple(fields), v);
  }

  Expr VisitExpr_(const TupleGetItemNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    return Compound(e, TupleGetItem(VisitExpr(t->tuple), t->index), v);
  }

  Expr VisitExpr_(const RefCreateNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefCreate(VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const RefReadNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefRead(VisitExpr(r->ref)), v);
  }

  Expr VisitExpr_(const RefWriteNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefWrite(VisitExpr(r->ref), VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const IfNode* i, const Var& v) final {
    Expr e = GetRef<Expr>(i);
    Expr ret = If(VisitExpr(i->cond), GetSubScope(e, 1)->let_list->Get(VisitExpr(i->true_branch)),
                  GetSubScope(e, 2)->let_list->Get(VisitExpr(i->false_branch)));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const FunctionNode* f, const Var& v) final {
    Expr e = GetRef<Expr>(f);
    Expr ret;
    if (f->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
      ret = e;
    } else {
      ret = Function(f->params, GetSubScope(e, 0)->let_list->Get(VisitExpr(f->body)), f->ret_type,
                     f->type_params, f->attrs);
    }
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const LetNode* l, const Var& v) final {
    Expr e = GetRef<Expr>(l);
    VisitExpr(l->value, l->var);
    Expr ret = GetSubScope(e, 0)->let_list->Get(VisitExpr(l->body));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const RelayConstantNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Atomic(e, v);
  }

  Expr VisitExpr_(const VarNode* vn, const Var& v) final {
    Expr e = GetRef<Expr>(vn);
    return Atomic(e, v);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    return Atomic(gv, v);
  }

  Expr VisitExpr_(const OpNode* op, const Var& v) final {
    Expr e = GetRef<Expr>(op);
    return Atomic(e, v);
  }
};

}  // namespace to_a_normal_form

Expr ToANormalFormExpr(const Expr& expr) {
  tvm::support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, expr);
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
  return to_a_normal_form::Fill::ToANormalForm(expr, dg, &scopes.first);
}

IRModule ToANormalForm(IRModule m) {
  tvm::Map<GlobalVar, BaseFunc> updates;
  auto funcs = m->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(tvm::relay::attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToANormalFormExpr(e); },
                          Downcast<Function>(it.second));
    ICHECK_EQ(FreeVars(ret).size(), 0)
        << AsText(ret) << "should not has free vars: " << FreeVars(ret);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    m->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToANF: transformed" << std::endl << m;

  return m;
}

MNM_REGISTER_GLOBAL("mnm.pass_.ToANormalForm").set_body_typed(ToANormalForm);

}  // namespace pass
}  // namespace mnm
