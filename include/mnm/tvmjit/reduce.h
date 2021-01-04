/*!
 * Copyright (c) 2019 by Contributors
 * \file reduce.h
 * \brief Data structures for reduction operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>

namespace mnm {
namespace op {
namespace tvmjit {

struct SumAttrs : public tvm::AttrsNode<SumAttrs> {
  tvm::Array<tvm::Integer> axis;
  tvm::Array<tvm::Integer> keepdims;
  TVM_DECLARE_ATTRS(SumAttrs, "attrs.SumAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(keepdims);
  }
};

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
