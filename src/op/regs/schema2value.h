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
  return IntValue::make(DataType::Int(64), a);
}

inline value::Value Bool(bool a) {
  MNM_PRELUDE();
  return BoolValue::make(a);
}

inline value::Value Double(double a) {
  MNM_PRELUDE();
  return FloatValue::make(DataType::Float(64), a);
}

inline value::Value String(const std::string& a) {
  MNM_PRELUDE();
  return StringValue::make(a);
}

inline value::Value TupleInt(const std::vector<int64_t>& a) {
  MNM_PRELUDE();
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
  MNM_PRELUDE();
  Array<Value> ret;
  for (const auto i : a.value()) {
    ret.push_back(IntValue::make(i->dtype, i->value));
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
