/*!
 * Copyright (c) 2019 by Contributors
 * \file ir_ext.h
 * \brief Extension of TVM/Relay IR
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include <string>
#include "./ir.h"

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {
class Module;

class ModuleObj : public ir::Object {
 public:
  Map<GlobalVar, Function> functions;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("functions", &functions);
    v->Visit("global_var_map_", &global_var_map_);
  }
  void Add(const GlobalVar& var, const Function& func, bool update = false);
  Function Lookup(const GlobalVar& var) const;
  Function Lookup(const std::string& name) const;
  bool ContainGlobalVar(const std::string& name) const;
  GlobalVar GetGlobalVar(const std::string& str) const;

 public:
  static constexpr const char* _type_key = "mnm.ir.Module";
  MNM_FINAL_OBJECT(ModuleObj, ir::Object);

 private:
  /*! \brief A map from string names to global type variables (ADT names)
   * that ensures global uniqueness.
   */
  Map<String, GlobalVar> global_var_map_;
  friend class Module;
};

class Module : public ir::ObjectRef {
 public:
  static Module make(Map<GlobalVar, Function> functions);
  static Module Global();
  MNM_OBJECT_REF(Module, ir::ObjectRef, ModuleObj);
};

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Constant ******/
namespace mnm {
namespace ir {

using RelayConstantNode = tvm::relay::ConstantNode;
using RelayConstant = tvm::relay::Constant;
using Constant = tvm::relay::Constant;

class ConstantNode : public RelayConstantNode {
 public:
  ObjectRef value{nullptr};

  bool SEqualReduce(const ConstantNode* other, tvm::SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(value);
  }
};

ObjectPtr<ConstantNode> MakeConstantNode(ObjectRef node_ref);
RelayConstant MakeConstant(ObjectRef node_ref);

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Var ******/
namespace mnm {
namespace ir {

class ExtendedVarNode : public VarNode {
 public:
  /*! \brief A hint for inplace write into may_share */
  mutable Var may_share{Var()};
};

/*!
 * \brief Create an extended meta variable
 * \param name_hint name_hint
 * \param type_annotation type_annotation
 * \param may_share the var with which it may share memory
 * \return a var which contains a pointer to ExtendedVarNode
 */
Var MakeVar(const std::string& name_hint, Type type_annotation, Var may_share = {});

}  // namespace ir
}  // namespace mnm
