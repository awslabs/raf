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

// GradTape stores the information necessary for doing imperative auto differentiation
//  * requires_grad     <=> tape.defined()
//  * retain_grad       <=> tape.defined() && tape->retain_grad
//  * (unused) is_leaf  <=> !tape.defined() || tape->prev_tapes.empty()
class GradTapeObj : public ir::Object {
 public:
  ir::Var grad;
  mutable value::ClosureValue bp;
  mutable ir::Array<ir::ObjectRef> prev_tapes;
  mutable bool retain_grad{false};
  static constexpr const char* _type_key = "mnm.binding.GradTapeObj";
  MNM_FINAL_OBJECT(GradTapeObj, ir::Object);
};

class GradTape : public ir::ObjectRef {
 public:
  static GradTape make(ir::Var grad, value::ClosureValue bp, ir::Array<ir::ObjectRef> prev_tapes);
  MNM_OBJECT_REF(GradTape, ir::ObjectRef, GradTapeObj);
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
  mutable value::Value value;
  mutable GradTape tape;
  static constexpr const char* _type_key = "mnm.binding.NDArrayBinding";
  MNM_FINAL_OBJECT(NDArrayBindingObj, BindingEntryObj);
};

class NDArrayBinding : public BindingEntry {
 public:
  static NDArrayBinding make(value::Value value, GradTape tape);
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

ir::Var BindNDArray(value::Value value, GradTape tape = {}, std::string name_hint = "");
ir::Var BindSymbol(ir::Expr expr, std::string name_hint = "");
BindingEntry LookupBinding(const ir::VarNode *var);

}  // namespace binding
}  // namespace mnm
