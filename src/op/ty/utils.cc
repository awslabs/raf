/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/utils.cc
 * \brief Typing utils
 */
#include "./utils.h"
#include "mnm/value_functor.h"

namespace mnm {
namespace op {
namespace type {

using namespace value;
using namespace tvm;

class ValueTyper : public ValueFunctor<Type(const Value&)> {
  Type VisitValue_(const IntValueObj* value) override {
    return TensorType::Scalar(DataType::Int(64));
  }

  Type VisitValue_(const FloatValueObj* value) override {
    return TensorType::Scalar(DataType::Float(64));
  }

  Type VisitValue_(const BoolValueObj* value) override {
    return TensorType::Scalar(DataType::Bool());
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
};

Type GetType(Value value) {
  if (!value.defined()) {
    return Type();
  }
  ValueTyper typer;
  return typer(value);
}

bool TypeCheck(const tvm::PrimExpr& cond) {
  using namespace tvm;
  if (const int64_t* pdiff = tir::as_const_int(cond)) {
    return pdiff[0];
  }
  return true;
}

}  // namespace type
}  // namespace op
}  // namespace mnm
