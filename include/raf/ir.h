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

#define RAF_BASE_OBJECT(TypeName, ParentType)                                                  \
  static uint32_t RuntimeTypeIndex() {                                                         \
    if (TypeName::_type_index != ::tvm::runtime::TypeIndex::kDynamic) {                        \
      return TypeName::_type_index;                                                            \
    }                                                                                          \
    return _GetOrAllocRuntimeTypeIndex();                                                      \
  }                                                                                            \
  static uint32_t _GetOrAllocRuntimeTypeIndex() {                                              \
    static uint32_t tidx = GetOrAllocRuntimeTypeIndex(                                         \
        TypeName::_type_key, TypeName::_type_index, ParentType::_GetOrAllocRuntimeTypeIndex(), \
        TypeName::_type_child_slots, TypeName::_type_child_slots_can_overflow);                \
    return tidx;                                                                               \
  }

#define RAF_FINAL_OBJECT(TypeName, ParentType)      \
  static const constexpr bool _type_final = true;   \
  static const constexpr int _type_child_slots = 0; \
  RAF_BASE_OBJECT(TypeName, ParentType)

#define RAF_OBJECT_REF(TypeName, ParentType, ObjectName)                                   \
  TypeName() {                                                                             \
  }                                                                                        \
  explicit TypeName(::tvm::runtime::ObjectPtr<::tvm::runtime::Object> n) : ParentType(n) { \
  }                                                                                        \
  ObjectName* operator->() const {                                                         \
    return static_cast<ObjectName*>(data_.get());                                          \
  }                                                                                        \
  using ContainerType = ObjectName;

#define RAF_NOTNULLABLE_OBJECT_REF(TypeName, ParentType, ObjectName)                       \
  explicit TypeName(::tvm::runtime::ObjectPtr<::tvm::runtime::Object> n) : ParentType(n) { \
  }                                                                                        \
  ObjectName* operator->() const {                                                         \
    return static_cast<ObjectName*>(data_.get());                                          \
  }                                                                                        \
  static constexpr bool _type_is_nullable = false;                                         \
  using ContainerType = ObjectName;

#define RAF_REGISTER_OBJECT_NO_REFLECT(TypeName)                              \
  static DMLC_ATTRIBUTE_UNUSED uint32_t __make_Object_tidx##_##TypeName##__ = \
      TypeName::_GetOrAllocRuntimeTypeIndex()

#define RAF_REGISTER_OBJECT_REFLECT(TypeName)                                                    \
  RAF_REGISTER_OBJECT_NO_REFLECT(TypeName);                                                      \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::ReflectionVTable::Registry& __make_Node##_##TypeName##__ = \
      ::tvm::ReflectionVTable::Global()                                                          \
          ->Register<TypeName, ::tvm::detail::ReflectionTrait<TypeName>>()                       \
          .set_creator(                                                                          \
              [](const std::string&) -> ::tvm::runtime::ObjectPtr<::tvm::runtime::Object> {      \
                return ::tvm::runtime::make_object<TypeName>();                                  \
              })

#include "./ir_ext.h"
#include "./dataflow_pattern.h"
