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
