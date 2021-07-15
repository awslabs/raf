/*!
 * Copyright (c) 2021 by Contributors
 * \file unary.h
 * \brief TVM attributes for unary operators
 */
#pragma once
#include <tvm/ir/attrs.h>

namespace mnm {
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
}  // namespace mnm
