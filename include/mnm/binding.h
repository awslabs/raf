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

// ADInfo stores the information necessary for doing imperative auto differentiation
class ADInfoObj : public ir::Object {
 public:
  value::ClosureValue bp;
  ir::Var ograd;
  ir::Array<ir::ObjectRef> inputs;
  bool retain_grad{false};
  static constexpr const char* _type_key = "mnm.binding.ADInfoObj";
  MNM_FINAL_OBJECT(ADInfoObj, ir::Object);
};

class ADInfo : public ir::ObjectRef {
 public:
  static ADInfo make(value::ClosureValue bp, ir::Var ograd, ir::Array<ir::ObjectRef> inputs);
  MNM_OBJECT_REF(ADInfo, ir::ObjectRef, ADInfoObj);
};

// BindingEntry stores auxiliary information for vars
class BindingEntryObj : public ir::Object {
 public:
  static constexpr const char* _type_key = "mnm.binding.Binding";
  MNM_BASE_OBJECT(BindingEntryObj, ir::Object);
};

class BindingEntry : public ir::ObjectRef {
 public:
  MNM_OBJECT_REF(BindingEntry, ir::ObjectRef, BindingEntryObj);
};

// NDArray's binding entry
class NDArrayBindingObj : public BindingEntryObj {
 public:
  value::Value value;
  mutable ADInfo ad_info;
  static constexpr const char* _type_key = "mnm.binding.NDArrayBinding";
  MNM_FINAL_OBJECT(NDArrayBindingObj, BindingEntryObj);
};

class NDArrayBinding : public BindingEntry {
 public:
  static NDArrayBinding make(value::Value value, ADInfo ad_info);
  MNM_OBJECT_REF(NDArrayBinding, BindingEntry, NDArrayBindingObj);
};

// Symbol's binding entry
class SymbolBindingObj : public BindingEntryObj {
 public:
  ir::Expr expr;
  static constexpr const char* _type_key = "mnm.binding.SymbolBinding";
  MNM_FINAL_OBJECT(SymbolBindingObj, BindingEntryObj);
};

class SymbolBinding : public BindingEntry {
 public:
  static SymbolBinding make(ir::Expr expr);
  MNM_OBJECT_REF(SymbolBinding, BindingEntry, SymbolBindingObj);
};

ir::Var BindNDArray(value::Value value, std::string name_hint = "", ADInfo ad_info = {});
ir::Var BindSymbol(ir::Expr expr, std::string name_hint = "");
BindingEntry LookupBinding(const ir::VarNode *var);

}  // namespace binding
}  // namespace mnm
