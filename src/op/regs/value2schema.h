/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/regs/value2schema.h
 * \brief Converters from values to RAF operator schemas
 */
#pragma once
#include <string>
#include <vector>
#include "raf/value.h"
#include "./regs_utils.h"

namespace raf {
namespace op {
namespace regs {
namespace value2schema {

using raf::ir::Array;

#define RAF_PRELUDE_ALLOW_NULL() \
  using namespace raf::value;    \
  using namespace raf::ir;       \
  if (!a.defined()) {            \
    return {};                   \
  }

#define RAF_PRELUDE_DISALLOW_NULL(type)                                             \
  using namespace raf::value;                                                       \
  using namespace raf::ir;                                                          \
  if (!a.defined()) {                                                               \
    LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\""             \
               << " is undefined (NULL), but is required to be of type " << (type); \
    throw;                                                                          \
  }

inline value::Value ArrayLike(const value::Value& a) {
  RAF_PRELUDE_ALLOW_NULL();
  if (a->IsInstance<IntValueObj>() || a->IsInstance<FloatValueObj>() ||
      a->IsInstance<BoolValueObj>() || a->IsInstance<BaseTensorValueObj>() ||
      a->IsInstance<TupleValueObj>() || a->IsInstance<VoidValueObj>() ||
      a->IsInstance<OpValueObj>() || a->IsInstance<ClosureValueObj>()) {
    return a;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not array-like";
  throw;
}

inline ir::Optional<value::Value> OptionalArrayLike(const value::Value& a) {
  if (!a.defined()) {
    return tvm::NullOpt;
  }
  return ArrayLike(a);
}

inline value::BaseTensorValue Tensor(const value::Value& a) {
  RAF_PRELUDE_ALLOW_NULL();
  if (const auto* v = a.as<BaseTensorValueObj>()) {
    return GetRef<BaseTensorValue>(v);
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a tensor";
  throw;
}

inline ir::Optional<value::BaseTensorValue> OptionalTensor(const value::Value& a) {
  if (!a.defined()) {
    return tvm::NullOpt;
  }
  return Tensor(a);
}

inline int64_t Int(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("an integer");
  if (const auto* v = a.as<IntValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer";
  throw;
}
inline bool Bool(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("boolean");
  if (const auto* v = a.as<BoolValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a bool value";
  throw;
}
inline double Double(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("double");
  if (const auto* v = a.as<FloatValueObj>()) {
    return v->value;
  }
  if (const auto* v = a.as<IntValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a double";
  throw;
}
inline std::string String(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("string");
  if (const auto* v = a.as<StringValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a string";
  throw;
}

inline std::vector<int64_t> TupleInt(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("tuple of integers");
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<int64_t> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<IntValueObj>()) {
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
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not tuple of integers";
  throw;
}

inline std::vector<int64_t> IntOrTupleInt(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("an integer or tuple of integers");
  if (const auto* v = a.as<IntValueObj>()) {
    return {v->value};
  }
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<int64_t> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<IntValueObj>()) {
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
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer or tuple of integers";
  throw;
}

inline ir::Optional<Array<value::IntValue>> IntArray(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("array of integers");
  if (const auto* v = a.as<IntValueObj>()) {
    return Array<value::IntValue>{value::IntValue::make(v->dtype, v->value)};
  }
  if (const auto* v = a.as<value::TupleValueObj>()) {
    Array<value::IntValue> ret;
    ret.reserve(v->fields.size());
    for (const tvm::runtime::ObjectRef& i : v->fields) {
      if (const auto* e = i.as<value::IntValueObj>()) {
        ret.push_back(value::IntValue::make(e->dtype, e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  } else if (auto* v = a.as<value::TensorTypeValueObj>()) {
    return tvm::NullOpt;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer or tuple of integers";
  throw;
}

inline std::vector<value::BaseTensorValue> TupleTensor(const value::Value& a) {
  RAF_PRELUDE_DISALLOW_NULL("tuple of tensors");
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<BaseTensorValue> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<BaseTensorValueObj>()) {
        ret.push_back(Downcast<BaseTensorValue>(i));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of tensors, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not tuple of tensors";
  throw;
}

#undef RAF_PRELUDE_DISALLOW_NULL
#undef RAF_PRELUDE_ALLOW_NULL

}  // namespace value2schema
}  // namespace regs
}  // namespace op
}  // namespace raf
