/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file type.h
 * \brief Type system
 */
#pragma once
#include <tvm/ir/type.h>
#include <tvm/ir/env_func.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <string>

#include "raf/value.h"
#include "raf/op.h"
#include "./registry.h"

namespace raf {
namespace ir {

using tvm::PrimType;
using tvm::PrimTypeNode;
using tvm::VoidType;
using tvm::relay::FuncType;
using tvm::relay::FuncTypeNode;
using tvm::relay::IncompleteType;
using tvm::relay::IncompleteTypeNode;
using tvm::relay::TensorType;
using tvm::relay::TensorTypeNode;
using tvm::relay::TupleType;
using tvm::relay::TupleTypeNode;
using tvm::relay::Type;
using tvm::relay::TypeNode;

using TypeInferenceFn = tvm::TypedEnvFunc<Type(const op::CallValues& value)>;

using OpType = FuncType;

/*!
 * \brief User defined type inference, it is used for inference from input types to output types
 *
 * \sa TypeInference
 */
class TypeInferenceNode : public tvm::TypeConstraintNode {
 public:
  /*!
   * \brief The function which takes input types and gives output types
   */
  TypeInferenceFn func;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_func", &func);
  }

  bool SEqualReduce(const TypeInferenceNode* other, tvm::SEqualReducer equal) const {
    return equal(func, other->func);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(func);
  }

  static constexpr const char* _type_key = "TypeInference";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeInferenceNode, TypeConstraintNode);
};

/*!
 * \brief Managed reference to TypeInferenceNode.
 * \sa TypeInferenceNode
 */
class TypeInference : public tvm::TypeConstraint {
 public:
  explicit TypeInference(TypeInferenceFn func);
  TVM_DEFINE_OBJECT_REF_METHODS(TypeInference, TypeConstraint, TypeInferenceNode);
};

OpType MakeOpType(const std::string& op_name, const std::string& fn_name,
                  tvm::runtime::TypedPackedFunc<Type(const op::CallValues& value)> fn);

}  // namespace ir
}  // namespace raf

#define RAF_OP_TYPE(op_name, fn_name, body)               \
  RELAY_REGISTER_OP(op_name).set_attr<::raf::ir::OpType>( \
      "OpType", ::raf::ir::MakeOpType(op_name, fn_name, body))
