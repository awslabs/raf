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
 * \file src/op/ty/utils.cc
 * \brief Typing utils
 */
#include "./utils.h"
#include "mnm/value_functor.h"
#include "mnm/pass.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;

class ValueTyper : public ValueFunctor<Type(const Value&)> {
  Type VisitValue_(const IntValueObj* value) override {
    return TensorType::Scalar(value->dtype);
  }

  Type VisitValue_(const FloatValueObj* value) override {
    return TensorType::Scalar(value->dtype);
  }

  Type VisitValue_(const BoolValueObj* value) override {
    return TensorType::Scalar(value->dtype);
  }

  Type VisitValue_(const StringValueObj* value) override {
    // fake type info
    return TensorType::Scalar(DataType::Int(64));
  }

  Type VisitValue_(const NoGradValueObj* value) override {
    // fake type info
    return TensorType::Scalar(DataType::Int(64));
  }

  Type VisitValue_(const TensorValueObj* value) override {
    const DLTensor* x = GetRef<Value>(value);
    auto shape = std::vector<Integer>(x->shape, x->shape + x->ndim);
    return TensorType({shape.begin(), shape.end()}, DataType(x->dtype));
  }

  Type VisitValue_(const TupleValueObj* value) override {
    std::vector<Type> fields;
    fields.reserve(value->fields.size());
    for (const auto& v : value->fields) {
      fields.push_back(VisitValue(v));
    }
    return TupleType(Array<Type>(fields.begin(), fields.end()));
  }

  Type VisitValue_(const TensorTypeValueObj* value) override {
    return value->type;
  }

  Type VisitValue_(const ClosureValueObj* value) override {
    std::vector<Type> fields;
    if (value->func->checked_type_.defined()) {
      return value->func->checked_type();
    }
    ir::Expr func = pass::InferType(value->func);
    return func->checked_type();
  }
};

Type GetType(Value value) {
  if (!value.defined()) {
    return Type();
  }
  ValueTyper typer;
  return typer(value);
}

bool TypeCheck(const PrimExpr& cond) {
  if (const int64_t* pdiff = tvm::tir::as_const_int(cond)) {
    return pdiff[0];
  }
  return true;
}

}  // namespace op
}  // namespace mnm
