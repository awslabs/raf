/*!
 * Copyright (c) 2019 by Contributors
 * \file ir_ext.h
 * \brief Extension of TVM/Relay IR
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include <string>
#include "./ir.h"

/****** mnm::ir::IRModule ******/
namespace mnm {
namespace ir {

IRModule GlobalModule();

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

  /*!
   * \brief Check if the constant is tensor
   * \return Whether the constant is tensor
   */
  bool IsTensor() const;

  bool SEqualReduce(const ConstantNode* other, tvm::SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(value);
  }
};

ObjectPtr<ConstantNode> MakeConstantNode(ObjectRef node_ref);
RelayConstant MakeConstant(ObjectRef node_ref);
RelayConstant MakeNull();
ObjectRef ConstantExtractValue(RelayConstant _node);

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

String AsText(const ObjectRef& node, bool show_meta_data = false);

}  // namespace ir
}  // namespace mnm
