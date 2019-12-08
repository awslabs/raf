/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/binding.cc
 * \brief Frontend-defined varioble-expression-value bindings
 */
#include "mnm/binding.h"
#include "mnm/registry.h"

namespace mnm {
namespace binding {
namespace {
MNM_REGISTER_OBJECT_REFLECT(BindingEntryObj);
}  // namespace

using namespace mnm::ir;
using namespace mnm::value;

class BindingMgr {
 public:
  std::mutex mu;
  std::unordered_map<const VarNode*, BindingEntry> bindings;

  static BindingMgr* Get() {
    static BindingMgr* instance = new BindingMgr();
    return instance;
  }
};

class BoundVarObj : public VarNode {
  // This is basically relay::VarNode, but with a customized callback that
  // deletes the weak reference inside BindingMgr
 public:
  ~BoundVarObj() {
    static BindingMgr* mgr = BindingMgr::Get();
    BindingEntry entry{nullptr};
    {
      std::lock_guard<std::mutex> lock(mgr->mu);
      auto iter = mgr->bindings.find(this);
      CHECK(iter != mgr->bindings.end());
      entry = iter->second;
      mgr->bindings.erase(iter);
    }
    // "entry" is destroyed here, to avoid potential recursive lock
  }
  static Var make(const std::string& name_hint) {
    ObjectPtr<BoundVarObj> n = make_object<BoundVarObj>();
    ObjectPtr<IdNode> id_ptr = make_object<IdNode>();
    id_ptr->name_hint = name_hint;
    n->vid = Id(id_ptr);
    return Var(n);
  }
};

BindingEntry BindingEntry::make(Expr expr, Value value) {
  ObjectPtr<BindingEntryObj> n = make_object<BindingEntryObj>();
  n->expr = std::move(expr);
  n->value = std::move(value);
  return BindingEntry(n);
}

Var BindExprValue(Expr expr, Value value, std::string name_hint) {
  static BindingMgr* mgr = BindingMgr::Get();
  static auto& bindings = mgr->bindings;
  Var var = BoundVarObj::make(name_hint);
  const VarNode* var_ptr = var.operator->();
  BindingEntry entry = BindingEntry::make(expr, value);
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    bindings.emplace(var_ptr, entry);
  }
  return var;
}

BindingEntry LookupBinding(const VarNode* var) {
  static BindingMgr* mgr = BindingMgr::Get();
  static const auto& bindings = mgr->bindings;
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    auto iter = bindings.find(var);
    return iter != bindings.end() ? iter->second : NullValue<BindingEntry>();
  }
}

Var BindConstValue(Value value, std::string name_hint) {
  return BindExprValue(MakeConstant(value), value, name_hint);
}

Expr LookupBoundExpr(Var var) {
  BindingEntry entry = LookupBinding(var.operator->());
  return entry.defined() ? entry->expr : NullValue<Expr>();
}

Value LookupBoundValue(ir::Var var) {
  BindingEntry entry = LookupBinding(var.operator->());
  return entry.defined() ? entry->value : NullValue<Value>();
}

class LetListExtractor final : public ExprVisitor {
 public:
  void StashVar(const Expr& expr) {
    if (const VarNode* var = expr.as<VarNode>()) {
      if (++in_degree[var] == 1) {
        queue.push_back(var);
      }
      if (out_edge != nullptr) {
        out_edge->push_back(var);
      }
    } else if (!expr->IsInstance<ConstantNode>()) {
      LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
    }
  }

  void VisitExpr_(const VarNode* var) final {
    LOG(FATAL) << "Should not be here";
  }

  void VisitExpr_(const TupleNode* node) final {
    for (const Expr& expr : node->fields) {
      StashVar(expr);
    }
  }

  void VisitExpr_(const CallNode* node) final {
    for (const Expr& expr : node->args) {
      StashVar(expr);
    }
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    StashVar(node->tuple);
  }

  std::vector<const VarNode*> queue;
  std::unordered_map<const VarNode*, int> in_degree;
  std::unordered_map<const VarNode*, const ExprNode*> bindings;
  std::unordered_map<const VarNode*, std::vector<const VarNode*> > graph;
  std::vector<const VarNode*>* out_edge = nullptr;

  Function Run(const Var& var, const Array<ObjectRef>& _params) {
    Array<Var> args;
    std::unordered_set<const Object*> arg_exists;
    for (const ObjectRef& p : _params) {
      args.push_back(Downcast<Var>(p));
      arg_exists.insert(p.get());
    }
    StashVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const Expr& expr = LookupBoundExpr(GetRef<Var>(var));
      const ExprNode* expr_node = bindings[var] = expr.operator->();
      queue.pop_back();
      out_edge = &graph[var];
      if (!arg_exists.count(var)) {
        CHECK(expr_node != nullptr);
        ExprVisitor::VisitExpr(expr);
      }
    }
    Expr body = var;
    queue.clear();
    queue.push_back(var.operator->());
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const ExprNode* expr = bindings[var];
      queue.pop_back();
      if (!arg_exists.count(var)) {
        CHECK(expr != nullptr);
        body = LetNode::make(GetRef<Var>(var), GetRef<Expr>(expr), body);
      }
      for (const VarNode* out : graph[var]) {
        if (--in_degree[out] == 0) {
          queue.push_back(out);
        }
      }
    }
    return FunctionNode::make(args, body, {}, {});
  }
};

Function ExtractLetList(Var var, Array<ObjectRef> params) {
  return LetListExtractor().Run(var, params);
}

Var GetGrad(Var var) {
  BindingEntry entry = LookupBinding(var.operator->());
  return entry.defined() ? entry->grad : NullValue<Var>();
}

void SetRequiresGrad(Var var, bool true_or_false) {
  BindingEntry entry = LookupBinding(var.operator->());
  CHECK(entry.defined());
  if (entry->grad.defined() == true_or_false) {
    return;
  }
  if (true_or_false) {
    std::string name = var->vid->name_hint;
    if (name != "") {
      name = name + ".grad";
    }
    entry->grad = BindExprValue(NullValue<Expr>(), NullValue<Value>(), name);
  } else {
    entry->grad = NullValue<Var>();
  }
}

MNM_REGISTER_GLOBAL("mnm.binding.BindConstValue").set_body_typed(BindConstValue);
MNM_REGISTER_GLOBAL("mnm.binding.BindExprValue").set_body_typed(BindExprValue);
MNM_REGISTER_GLOBAL("mnm.binding.LookupBoundExpr").set_body_typed(LookupBoundExpr);
MNM_REGISTER_GLOBAL("mnm.binding.LookupBoundValue").set_body_typed(LookupBoundValue);
MNM_REGISTER_GLOBAL("mnm.binding.ExtractLetList").set_body_typed(ExtractLetList);
MNM_REGISTER_GLOBAL("mnm.binding.GetGrad").set_body_typed(GetGrad);
MNM_REGISTER_GLOBAL("mnm.binding.SetRequiresGrad").set_body_typed(SetRequiresGrad);

}  // namespace binding
}  // namespace mnm
