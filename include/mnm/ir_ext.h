/*!
 * Copyright (c) 2019 by Contributors
 * \file ir_ext.h
 * \brief Extension of TVM/Relay IR
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include "./ir.h"

/****** mnm::ir::Module ******/
namespace mnm {
namespace ir {

class ModuleObj : public ir::Object {
 public:
  Map<GlobalVar, Function> functions;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("functions", &functions);
  }
  void Add(const GlobalVar& var, const Function& func);
  Function Lookup(const GlobalVar& var) const;

 public:
  static constexpr const char* _type_key = "mnm.ir.Module";
  MNM_FINAL_OBJECT(ModuleObj, ir::Object);
};

class Module : public ObjectRef {
 public:
  static Module make(Map<GlobalVar, Function> functions);
  static Module Global();
  MNM_OBJECT_REF(Module, ObjectRef, ModuleObj);
};

}  // namespace ir
}  // namespace mnm

/****** mnm::ir::Module ******/
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

RelayConstant MakeConstant(ObjectRef node_ref);

}  // namespace ir
}  // namespace mnm
