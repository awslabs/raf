/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/schema2value.h
 * \brief Converters from MNM operator schemas to values
 */
#pragma once
#include <string>
#include <vector>
#include <utility>
#include "mnm/value.h"

namespace mnm {
namespace op {
namespace regs {
namespace schema2value {

#define MNM_PRELUDE()         \
  using namespace mnm::value; \
  using namespace mnm::ir;

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
  MNM_PRELUDE();
  return IntValue::make(a);
}

inline value::Value Bool(bool a) {
  MNM_PRELUDE();
  return BoolValue::make(a);
}

inline value::Value Double(double a) {
  MNM_PRELUDE();
  return FloatValue::make(a);
}

inline value::Value String(const std::string& a) {
  MNM_PRELUDE();
  return StringValue::make(a);
}

inline value::Value TupleInt(const std::vector<int64_t>& a) {
  MNM_PRELUDE();
  Array<Value> ret;
  for (const auto i : a) {
    ret.push_back(IntValue::make(i));
  }
  return TupleValue::make(std::move(ret));
}

inline value::Value IntOrTupleInt(const std::vector<int64_t>& a) {
  return TupleInt(a);
}

inline value::Value IntArray(const ir::Optional<ir::Array<value::IntValue>> a) {
  MNM_PRELUDE();
  Array<Value> ret;
  for (const auto i : a.value()) {
    ret.push_back(IntValue::make(i->data));
  }
  return TupleValue::make(std::move(ret));
}

inline value::Value TupleTensor(const std::vector<value::BaseTensorValue>& a) {
  MNM_PRELUDE();
  Array<Value> ret;
  for (const auto i : a) {
    ret.push_back(i);
  }
  return TupleValue::make(std::move(ret));
}

#undef MNM_PRELUDE

}  // namespace schema2value
}  // namespace regs
}  // namespace op
}  // namespace mnm
