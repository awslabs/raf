/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ir.h
 * \brief A compatibility layer between RAF and TVM/Relay IR.
 */
#pragma once
#include <string>
#include "tvm/ir/module.h"
#include "tvm/runtime/object.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/container/string.h"
#include "tvm/node/node.h"
#include "tvm/relay/base.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/expr_functor.h"
#include "tvm/relay/op.h"
#include "tvm/relay/op_attr_types.h"
#include "tvm/relay/type.h"
#include "tvm/relay/dataflow_pattern.h"
#include "tvm/relay/dataflow_matcher.h"
#include "tvm/ir/op.h"

namespace tvm {
namespace relay {

extern bool IsDynamic(const Type& ty);
extern Expr ToTupleType(const Type& ty, const std::vector<Expr>& exprs);

}  // namespace relay
}  // namespace tvm
namespace raf {
namespace ir {

// Containers
using tvm::Array;
using tvm::ArrayNode;
using tvm::Map;
using tvm::MapNode;
using tvm::Optional;
using tvm::String;

// Scalars
using tvm::Bool;
using tvm::FloatImm;
using tvm::FloatImmNode;
using tvm::Integer;
using tvm::IntImm;
using tvm::IntImmNode;
using tvm::PrimExpr;

// Attributes
using tvm::Attrs;
using tvm::AttrsNode;

// TVM basic objects
using tvm::NullValue;
using tvm::Target;

// TVM runtime objects
using tvm::runtime::DataType;
using tvm::runtime::Downcast;
using tvm::runtime::GetObjectPtr;
using tvm::runtime::GetRef;
using tvm::runtime::make_object;
using tvm::runtime::NDArray;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::ObjectPtrEqual;
using tvm::runtime::ObjectPtrHash;
using tvm::runtime::ObjectRef;
using tvm::runtime::SaveDLTensor;
using tvm::runtime::String2DLDataType;
using tvm::runtime::TypeIndex;

// TVM IRModule
using tvm::BaseFunc;
using tvm::IRModule;
using tvm::IRModuleNode;

// Relay Expression
using tvm::relay::Expr;
using tvm::relay::ExprNode;
using tvm::relay::IndexExpr;

using tvm::relay::Id;
using tvm::relay::IdNode;

using tvm::Op;
using tvm::OpNode;

using tvm::relay::Tuple;
using tvm::relay::TupleNode;

using tvm::relay::Var;
using tvm::relay::VarNode;

using tvm::relay::GlobalVar;
using tvm::relay::GlobalVarNode;

using tvm::relay::Function;
using tvm::relay::FunctionNode;

using tvm::relay::Call;
using tvm::relay::CallNode;

using tvm::relay::Let;
using tvm::relay::LetNode;

using tvm::relay::If;
using tvm::relay::IfNode;

using tvm::relay::TupleGetItem;
using tvm::relay::TupleGetItemNode;

using tvm::relay::RefCreate;
using tvm::relay::RefCreateNode;

using tvm::relay::RefRead;
using tvm::relay::RefReadNode;

using tvm::relay::RefWrite;
using tvm::relay::RefWriteNode;

using tvm::relay::TempExpr;
using tvm::relay::TempExprNode;

// Relay Types
using tvm::relay::Any;
using tvm::relay::AnyNode;
using tvm::relay::Kind;

using tvm::relay::Type;
using tvm::relay::TypeNode;

using tvm::relay::TensorType;
using tvm::relay::TensorTypeNode;

using tvm::relay::TypeVar;
using tvm::relay::TypeVarNode;

using tvm::relay::GlobalTypeVar;
using tvm::relay::GlobalTypeVarNode;

using tvm::relay::TypeCall;
using tvm::relay::TypeCallNode;

using tvm::relay::IncompleteType;
using tvm::relay::IncompleteTypeNode;

using tvm::relay::FuncType;
using tvm::relay::FuncTypeNode;

using tvm::relay::TupleType;
using tvm::relay::TupleTypeNode;

using tvm::relay::TypeConstraint;
using tvm::relay::TypeConstraintNode;

using tvm::relay::TypeRelation;
using tvm::relay::TypeRelationNode;

using tvm::relay::TypeReporter;

// Relay Functors
using tvm::relay::ExprFunctor;
using tvm::relay::ExprMutator;
using tvm::relay::ExprRewriter;
using tvm::relay::ExprVisitor;
using tvm::relay::MixedModeMutator;
using tvm::relay::MixedModeVisitor;

// Relay attributes
using tvm::WithAttr;

namespace attr {
using tvm::relay::attr::kClosure;
using tvm::relay::attr::kCompiler;
using tvm::relay::attr::kComposite;
using tvm::relay::attr::kInline;
using tvm::relay::attr::kPartitionedFromPattern;
using tvm::relay::attr::kPrimitive;
using tvm::relay::attr::kSkipOptimization;
/*! \brief Mark the dialect of the function. */
constexpr const char* kDialect = "Dialect";
/*! \brief Mark the fusion pattern name. */
constexpr const char* kPatternName = "PatternName";
}  // namespace attr

}  // namespace ir
}  // namespace raf

#define RAF_BASE_OBJECT(TypeName, ParentType) TVM_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)

#define RAF_FINAL_OBJECT(TypeName, ParentType) TVM_DECLARE_FINAL_OBJECT_INFO(TypeName, ParentType)

#define RAF_FINAL_OBJECT_NOCHECK(TypeName, ParentType)                                         \
  static const constexpr bool _type_final = true;                                              \
  static const constexpr int _type_child_slots = 0;                                            \
  static uint32_t RuntimeTypeIndex() {                                                         \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||    \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,             \
                  "Need to set _type_child_slots when parent specifies it.");                  \
    if (TypeName::_type_index != ::tvm::runtime::TypeIndex::kDynamic) {                        \
      return TypeName::_type_index;                                                            \
    }                                                                                          \
    return _GetOrAllocRuntimeTypeIndex();                                                      \
  }                                                                                            \
  static uint32_t _GetOrAllocRuntimeTypeIndex() {                                              \
    static uint32_t tindex = Object::GetOrAllocRuntimeTypeIndex(                               \
        TypeName::_type_key, TypeName::_type_index, ParentType::_GetOrAllocRuntimeTypeIndex(), \
        TypeName::_type_child_slots, TypeName::_type_child_slots_can_overflow);                \
    return tindex;                                                                             \
  }

#define RAF_OBJECT_REF(TypeName, ParentType, ObjectName) \
  TVM_DEFINE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)

#define RAF_MUTABLE_OBJECT_REF(TypeName, ParentType, ObjectName) \
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)

#define RAF_NOTNULLABLE_OBJECT_REF(TypeName, ParentType, ObjectName) \
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)

#define RAF_MUTABLE_NOTNULLABLE_OBJECT_REF(TypeName, ParentType, ObjectName) \
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)

#define RAF_REGISTER_OBJECT_NO_REFLECT(TypeName) TVM_REGISTER_OBJECT_TYPE(TypeName)

#define RAF_REGISTER_OBJECT_REFLECT(TypeName) TVM_REGISTER_NODE_TYPE(TypeName)
