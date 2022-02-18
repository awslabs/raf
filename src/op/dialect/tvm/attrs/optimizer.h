/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
