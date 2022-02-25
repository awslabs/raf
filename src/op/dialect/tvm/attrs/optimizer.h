/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file optimizer.h
 * \brief Extra TVM attributes for optimizer operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include "raf/ir.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;

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
}  // namespace raf
