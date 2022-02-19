/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
  /*!
   * \brief Check if the constant is a scalar
   * \return Whether the constant is a scalar
   */
  bool IsScalar() const;

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

/*!
 * \brief Extract the may_share field of an extended variable
 * \param var the variable
 * \return the may_share field of this variable
 */
Var GetMayShare(Expr var);

/*!
 * \brief Try to get the root may_share field of an extended variable. If the variable
 * does not have may_share, then returns itself.
 * \param var the variable
 * \return the may_share field of this variable, or the variable itself otherwise.
 */
Var TryGetMayShare(Expr var);

std::string AsText(const ObjectRef& node, bool show_meta_data = false);

}  // namespace ir
}  // namespace mnm
