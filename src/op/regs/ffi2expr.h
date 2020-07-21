/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/ffi2expr.h
 * \brief Converters from TVM FFI to Relay expressions
 */
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "./regs_utils.h"

namespace mnm {
namespace op {
namespace regs {
namespace ffi2expr {

#define MNM_PRELUDE()                                          \
  using namespace mnm::ir;                                     \
  using namespace mnm::value;                                  \
  using mnm::tensor::Tensor;                                   \
  int type_code = a.type_code();                               \
  if (type_code == kTVMObjectHandle && (a).IsObjectRef<Expr>()) { \
    return a.AsObjectRef<Expr>();                              \
  }

#define MNM_CONST(type, value) MakeConstant(type::make(value))

inline ir::Expr ArrayLike(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return MNM_CONST(IntValue, a.operator int64_t());
  }
  if (type_code == kDLFloat) {
    return MNM_CONST(FloatValue, a.operator double());
  }
  if (type_code == kTVMNullptr) {
    return MakeConstant(NullValue<Value>());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not array-like";
  throw;
}

inline ir::Expr Tensor(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kTVMNDArrayHandle || type_code == kTVMNullptr) {
    return MNM_CONST(TensorValue, a.operator tvm::runtime::NDArray());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not a tensor";
  throw;
}

inline ir::Expr Int(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return MNM_CONST(IntValue, a.operator int64_t());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer";
  throw;
}

inline ir::Expr Bool(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    int64_t value = a.operator int64_t();
    if (value == 0 || value == 1) {
      return MNM_CONST(BoolValue, static_cast<bool>(value));
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
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return MNM_CONST(FloatValue, a.operator int64_t());
  }
  if (type_code == kDLFloat) {
    return MNM_CONST(FloatValue, a.operator double());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not double";
  throw;
}

inline ir::Expr String(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kTVMStr) {
    return MNM_CONST(StringValue, a.operator std::string());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is a string";
  throw;
}

inline ir::Expr TupleInt(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<Value> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(IntValue::make(e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of integers, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return MNM_CONST(TupleValue, std::move(ret));
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not tuple of integers";
  throw;
}

inline ir::Expr IntOrTupleInt(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return MNM_CONST(TupleValue, tvm::Array<Value>({IntValue::make(a.operator int64_t())}));
  }
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    Array<Value> ret;
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(IntValue::make(e->value));
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
  MNM_PRELUDE();
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

#undef MNM_MAKE_CONST
#undef MNM_PRELUDE

}  // namespace ffi2expr
}  // namespace regs
}  // namespace op
}  // namespace mnm
