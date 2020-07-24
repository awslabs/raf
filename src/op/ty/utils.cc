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

class Typer : public ValueFunctor<Type(const Value&)> {
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
  Typer typer;
  return typer(value);
}

bool TypeCheckEqual(const tvm::PrimExpr& lhs, const tvm::PrimExpr& rhs) {
  using namespace tvm;
  if (lhs.as<tir::AnyNode>() || rhs.as<tir::AnyNode>()) {
    return true;
  }
  PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return true;
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
