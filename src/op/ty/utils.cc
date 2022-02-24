/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/utils.cc
 * \brief Typing utils
 */
#include "./utils.h"
#include "raf/value_functor.h"
#include "raf/pass.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;

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
}  // namespace raf
