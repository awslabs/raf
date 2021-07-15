/*!
 * Copyright (c) 2021 by Contributors
 * \file optimizer.h
 * \brief Extra TVM attributes for optimizer operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;

struct SgdAttrs : public tvm::AttrsNode<SgdAttrs> {
  double mu;
  double learning_rate;
  // declare attribute fields in header file
  TVM_DECLARE_ATTRS(SgdAttrs, "attrs.SgdAttrs") {
    TVM_ATTR_FIELD(mu);
    TVM_ATTR_FIELD(learning_rate);
  }
};

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
