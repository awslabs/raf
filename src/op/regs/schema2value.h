/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/regs/schema2value.h
 * \brief Converters from RAF operator schemas to values
 */
#pragma once
#include <string>
#include <vector>
#include <utility>
#include "raf/value.h"

namespace raf {
namespace op {
namespace regs {
namespace schema2value {

#define RAF_PRELUDE()         \
  using namespace raf::value; \
  using namespace raf::ir;

inline value::Value ArrayLike(const value::Value& a) {
  return a;
}

inline value::Value OptionalArrayLike(const ir::Optional<value::Value> a) {
  if (a.defined()) {
    return a.value();
  } else {
    return {};
  }
}

inline value::Value Tensor(const value::BaseTensorValue& a) {
  return a;
}

inline value::Value OptionalTensor(const ir::Optional<value::BaseTensorValue> a) {
  if (a.defined()) {
    return a.value();
  } else {
    return {};
  }
}

inline value::Value Int(int64_t a) {
  RAF_PRELUDE();
  return IntValue::make(DataType::Int(64), a);
}

inline value::Value Bool(bool a) {
  RAF_PRELUDE();
  return BoolValue::make(a);
}

inline value::Value Double(double a) {
  RAF_PRELUDE();
  return FloatValue::make(DataType::Float(64), a);
}

inline value::Value String(const std::string& a) {
  RAF_PRELUDE();
  return StringValue::make(a);
}

inline value::Value TupleInt(const std::vector<int64_t>& a) {
  RAF_PRELUDE();
  Array<Value> ret;
  for (const auto i : a) {
    ret.push_back(IntValue::make(DataType::Int(64), i));
  }
  return TupleValue::make(std::move(ret));
}

inline value::Value IntOrTupleInt(const std::vector<int64_t>& a) {
  return TupleInt(a);
}

inline value::Value IntArray(const ir::Optional<ir::Array<value::IntValue>> a) {
  RAF_PRELUDE();
  Array<Value> ret;
  for (const auto i : a.value()) {
    ret.push_back(IntValue::make(i->dtype, i->value));
  }
  return TupleValue::make(std::move(ret));
}

inline value::Value TupleTensor(const std::vector<value::BaseTensorValue>& a) {
  RAF_PRELUDE();
  Array<Value> ret;
  for (const auto i : a) {
    ret.push_back(i);
  }
  return TupleValue::make(std::move(ret));
}

#undef RAF_PRELUDE

}  // namespace schema2value
}  // namespace regs
}  // namespace op
}  // namespace raf
