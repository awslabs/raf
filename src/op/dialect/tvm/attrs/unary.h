/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file unary.h
 * \brief TVM attributes for unary operators
 */
#pragma once
#include <tvm/ir/attrs.h>

namespace raf {
namespace op {
namespace tvm_dialect {

struct UnaryDxAttr : public tvm::AttrsNode<UnaryDxAttr> {
  std::string grad_mode;

  TVM_DECLARE_ATTRS(UnaryDxAttr, "relay.attrs.UnaryDxAttr") {
    TVM_ATTR_FIELD(grad_mode).describe(
        "Indicate how to calculate the gradient: using input, output or both");
  }
};

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
