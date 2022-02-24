/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/regs/ffi2expr.h
 * \brief Converters from TVM FFI to Relay expressions
 */
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "raf/ir.h"
#include "raf/value.h"
#include "raf/registry.h"
#include "./regs_utils.h"
#include "tvm/runtime/c_runtime_api.h"

namespace raf {
namespace op {
namespace regs {
namespace ffi2expr {

#define RAF_PRELUDE()                                             \
  using namespace raf::ir;                                        \
  using namespace raf::value;                                     \
  using raf::tensor::Tensor;                                      \
  int type_code = a.type_code();                                  \
  if (type_code == kTVMObjectHandle && (a).IsObjectRef<Expr>()) { \
    return a.AsObjectRef<Expr>();                                 \
  }

#define RAF_CONST(type, value) MakeConstant(type::make(value))

inline ir::Expr ArrayLike(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    return RAF_CONST(ScalarValue, a.operator int64_t());
  }
  if (type_code == kDLFloat) {
    return RAF_CONST(ScalarValue, a.operator double());
  }
  if (type_code == kTVMNullptr) {
    return MakeConstant(NullValue<Value>());
  }
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    ir::Array<Value> fields;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        int64_t val = e->value;
        fields.push_back(IntValue::make(e->dtype, val));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of integers, "
                 << "because the " << ToOrdinal(fields.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return RAF_CONST(TupleValue, std::move(fields));
  }

  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not array-like";
  throw;
}

inline ir::Expr OptionalArrayLike(const registry::TVMArgValue& a) {
  return ArrayLike(a);
}

inline ir::Expr Tensor(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kTVMNDArrayHandle) {
    return RAF_CONST(TensorValue, a.operator tvm::runtime::NDArray());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not a tensor";
  throw;
}

inline ir::Expr OptionalTensor(const registry::TVMArgValue& a) {
  if (a.type_code() == kTVMNullptr) {
    return ir::MakeConstant(value::Value());
  }
  return Tensor(a);
}

inline ir::Expr Int(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    return RAF_CONST(ScalarValue, a.operator int64_t());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer";
  throw;
}

inline ir::Expr Bool(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    int64_t value = a.operator int64_t();
    if (value == 0 || value == 1) {
      return RAF_CONST(BoolValue, static_cast<bool>(value));
    }
    LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
               << "\" is not boolean. Its value is " << value;
    throw;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not boolean";
  throw;
}

inline ir::Expr Double(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    return MakeConstant(FloatValue::make(DataType::Float(64), a.operator int64_t()));
  }
  if (type_code == kDLFloat) {
    return MakeConstant(FloatValue::make(DataType::Float(64), a.operator double()));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not double";
  throw;
}

inline ir::Expr String(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kTVMStr) {
    return RAF_CONST(StringValue, a.operator std::string());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is a string";
  throw;
}

inline ir::Expr TupleInt(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<Value> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(IntValue::make(e->dtype, e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of integers, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return RAF_CONST(TupleValue, std::move(ret));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not tuple of integers";
  throw;
}

inline ir::Expr IntOrTupleInt(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    return RAF_CONST(TupleValue, tvm::Array<Value>({ScalarValue::make(a.operator int64_t())}));
  }
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<Value> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(IntValue::make(e->dtype, e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return MakeConstant(TupleValue::make(std::move(ret)));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer or tuple of integers";
  throw;
}

inline ir::Expr IntArray(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  if (type_code == kDLInt) {
    return RAF_CONST(TupleValue, tvm::Array<Value>({ScalarValue::make(a.operator int64_t())}));
  }
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<Value> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(IntValue::make(e->dtype, e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return MakeConstant(TupleValue::make(std::move(ret)));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer or tuple of integers";
  throw;
}

inline ir::Expr TupleTensor(const registry::TVMArgValue& a) {
  RAF_PRELUDE();
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<tvm::relay::Expr> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<VarNode>()) {
        ret.push_back(Downcast<Var>(i));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of tensors, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return binding::BindSymbol(tvm::relay::Tuple(ret));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not tuple of tensors";
  throw;
}

#undef RAF_MAKE_CONST
#undef RAF_PRELUDE

}  // namespace ffi2expr
}  // namespace regs
}  // namespace op
}  // namespace raf
