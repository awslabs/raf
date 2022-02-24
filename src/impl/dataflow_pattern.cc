/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/dataflow_pattern.cc
 * \brief Dataflow pattern of RAF IR
 */

#include <tvm/node/repr_printer.h>

#include "raf/registry.h"
#include "raf/dataflow_pattern.h"

namespace raf {
namespace ir {

using tvm::ReprPrinter;

// Pattern sugars
DFPattern IsRelayConstant() {
  return RelayConstantPattern(make_object<RelayConstantPatternNode>());
}

DFPattern IsConstant(ObjectRef value) {
  ObjectPtr<ConstantPatternNode> n = make_object<ConstantPatternNode>();
  n->value = std::move(value);
  return RelayConstantPattern(n);
}

RAF_REGISTER_GLOBAL("raf.ir.dataflow_pattern.is_constant").set_body_typed(IsConstant);

}  // namespace ir
}  // namespace raf
