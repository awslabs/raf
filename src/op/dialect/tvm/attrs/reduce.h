/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file reduce.h
 * \brief Extra TVM attributes for reduction operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include "raf/ir.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;

struct SumAttrs : public tvm::AttrsNode<SumAttrs> {
  Array<Integer> axis;
  Array<Integer> keepdims;
  bool exclude;
  TVM_DECLARE_ATTRS(SumAttrs, "attrs.SumAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(keepdims);
    TVM_ATTR_FIELD(exclude);
  }
};

struct MeanDxAttrs : public tvm::AttrsNode<MeanDxAttrs> {
  Array<Integer> axis;
  Array<Integer> shape;
  bool keepdims;
  bool exclude;

  TVM_DECLARE_ATTRS(MeanDxAttrs, "attrs.MeanDxAttrs") {
    TVM_ATTR_FIELD(axis);
    TVM_ATTR_FIELD(shape);
    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false).describe(
        "Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
