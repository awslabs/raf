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
MNM_REGISTER_GLOBAL("mnm.binding.GetGrad").set_body_typed(GetGrad);
MNM_REGISTER_GLOBAL("mnm.binding.SetRequiresGrad").set_body_typed(SetRequiresGrad);

}  // namespace binding
}  // namespace mnm
