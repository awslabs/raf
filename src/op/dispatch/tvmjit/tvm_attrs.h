/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/tvmjit/tvm_attrs.h
 * \brief Attributes defined in TVM
 */
#pragma once
#include "tvm/ir/attrs.h"
#include "tvm/relay/attrs/nn.h"
#include "tvm/relay/attrs/reduce.h"
#include "../../schema/ufunc.h"

namespace mnm {
namespace op {
namespace tvmjit {
namespace tvm_attrs {
using tvm::relay::BiasAddAttrs;
using tvm::relay::Conv2DAttrs;
using tvm::relay::ReduceAttrs;
}  // namespace tvm_attrs

using tvm::Attrs;
using namespace mnm::op::schema;

template <typename T>
std::vector<value::Value> BinarySchema2Args(const T* args) {
  return {args->x1, args->x2};
}

inline std::vector<std::string> BinarySchemaArgNames(const op::CallValues& call) {
  return {"x1", "x2"};
}

template <typename T>
Attrs GenericAttrs(const T* args) {
  return Attrs();
}

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
