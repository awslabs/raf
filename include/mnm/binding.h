/*!
 * Copyright (c) 2019 by Contributors
 * \file binding.h
 * \brief Frontend-defined varioble-expression-value bindings
 */
#pragma once
#include <string>
#include "./ir.h"
#include "./value.h"

namespace mnm {
namespace binding {

class BindingEntryObj final : public ir::Object {
 public:
  ir::Expr expr{nullptr};
  value::Value value{nullptr};
  mutable ir::Var grad{nullptr};
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  static constexpr const char* _type_key = "mnm.binding.BindingEntry";
  MNM_FINAL_OBJECT(BindingEntryObj, ir::Object);
};

class BindingEntry final : public ir::ObjectRef {
 public:
  static BindingEntry make(ir::Expr expr, value::Value value);
  MNM_OBJECT_REF(BindingEntry, ir::ObjectRef, BindingEntryObj);
};

ir::Var BindConstValue(value::Value value, std::string name_hint = "");
ir::Var BindExprValue(ir::Expr expr, value::Value value, std::string name_hint = "");
ir::Expr LookupBoundExpr(ir::Var var);
value::Value LookupBoundValue(ir::Var var);
}  // namespace binding
}  // namespace mnm
