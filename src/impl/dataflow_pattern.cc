/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/dataflow_pattern.cc
 * \brief Dataflow pattern of Meta IR
 */

#include <tvm/node/repr_printer.h>

#include "mnm/registry.h"
#include "mnm/dataflow_pattern.h"

namespace mnm {
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

MNM_REGISTER_GLOBAL("mnm.ir.dataflow_pattern.is_constant").set_body_typed(IsConstant);

}  // namespace ir
}  // namespace mnm
