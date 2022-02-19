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
 * \file reduce.h
 * \brief Extra TVM attributes for reduction operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;

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
}  // namespace mnm
