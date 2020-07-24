/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/ffi2schema.h
 * \brief Converters from TVM FFI to MNM operator schema
 */
#pragma once
#include <string>
#include <vector>
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/binding.h"
#include "./regs_utils.h"

namespace mnm {
namespace op {
namespace regs {
namespace ffi2schema {

#define MNM_PRELUDE()         \
  using namespace mnm::ir;    \
  using namespace mnm::value; \
  int type_code = a.type_code();

inline value::Value ArrayLike(const registry::TVMArgValue& a, binding::GradTape* tape) {
  MNM_PRELUDE();
  if (type_code == kTVMNullptr) {
    return {};
  }
  if (type_code == kTVMObjectHandle && a.IsObjectRef<Var>()) {
    using binding::NDArrayBindingObj;
    auto* bound = binding::LookupBinding(a.AsObjectRef<Var>().operator->()).as<NDArrayBindingObj>();
    *tape = bound->tape;
    return bound->value;
  }
  if (type_code == kDLInt) {
    return IntValue::make(a.operator int64_t());
  }
  if (type_code == kDLFloat) {
    return FloatValue::make(a.operator double());
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not array-like";
  throw;
}
inline value::BaseTensorValue Tensor(const registry::TVMArgValue& a, binding::GradTape* tape) {
  MNM_PRELUDE();
  if (type_code == kTVMObjectHandle && a.IsObjectRef<Var>()) {
    using binding::NDArrayBindingObj;
    auto* bound = binding::LookupBinding(a.AsObjectRef<Var>().operator->()).as<NDArrayBindingObj>();
    *tape = bound->tape;
    return Downcast<TensorValue>(bound->value);
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not a tensor";
  throw;
}
inline int64_t Int(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return a.operator int64_t();
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer";
  throw;
}
inline bool Bool(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    int64_t v = a;
    if (v == 0 || v == 1) {
      return v;
    }
    LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
               << "\" is not boolean, its value is " << v;
    throw;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not boolean";
  throw;
}
inline double Double(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLFloat) {
    return a.operator double();
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not a double";
  throw;
}
inline std::string String(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kTVMStr) {
    return a.operator std::string();
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not a string";
  throw;
}
inline std::vector<int64_t> TupleInt(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    std::vector<int64_t> ret;
    ret.reserve(n->size());
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(e->value);
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of integers, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not tuple of integers";
  throw;
}

inline std::vector<int64_t> IntOrTupleInt(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  if (type_code == kDLInt) {
    return {a.operator int64_t()};
  }
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    std::vector<int64_t> ret;
    ret.reserve(n->size());
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<IntImmNode>()) {
        ret.push_back(e->value);
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not an integer or tuple of integers";
  throw;
}
inline std::vector<value::BaseTensorValue> TupleTensor(const registry::TVMArgValue& a) {
  MNM_PRELUDE();
  const Object* _ptr = a.ptr<Object>();
  if (type_code == kTVMObjectHandle && _ptr->IsInstance<ArrayNode>()) {
    const ArrayNode* n = static_cast<const ArrayNode*>(_ptr);
    std::vector<BaseTensorValue> ret;
    ret.reserve(n->size());
    for (const ObjectRef& i : *n) {
      if (const auto* e = i.as<VarNode>()) {
        using binding::NDArrayBindingObj;
        auto* bound = binding::LookupBinding(e).as<NDArrayBindingObj>();
        ret.push_back(Downcast<TensorValue>(bound->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of tensors, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << GetTypeStr(a)
             << "\" is not tuple of tensors";
  throw;
}

#undef MNM_PRELUDE

}  // namespace ffi2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm
