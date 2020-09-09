/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/tvmjit/tvm_attrs.h
 * \brief Attributes defined in TVM
 */
#pragma once
#include "tvm/relay/attrs/nn.h"
#include "tvm/relay/attrs/reduce.h"

namespace mnm {
namespace op {
namespace tvmjit {
namespace tvm_attrs {
using tvm::relay::BiasAddAttrs;
using tvm::relay::Conv2DAttrs;
using tvm::relay::ReduceAttrs;
}  // namespace tvm_attrs
}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
