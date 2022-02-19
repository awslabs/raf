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
 * \file transform.h
 * \brief Extra TVM attributes for transform operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;

/*! \brief Attributes for StridedSlice operator */
struct StridedSliceDxAttrs : public tvm::AttrsNode<StridedSliceDxAttrs> {
  Optional<Array<Integer>> primal_shape;
  Optional<Array<Integer>> begin;
  Optional<Array<Integer>> end;
  Optional<Array<Integer>> strides;
  std::string slice_mode;

  TVM_DECLARE_ATTRS(StridedSliceDxAttrs, "relay.attrs.StridedSliceDxAttrs") {
    TVM_ATTR_FIELD(primal_shape).describe("Shape of the primal input to assist in backward pass");
    TVM_ATTR_FIELD(begin).describe("Indices for begin of slice, begin index is also inclusive");
    TVM_ATTR_FIELD(end).describe("Indices for end of slice, end index is exclusive");
    TVM_ATTR_FIELD(strides).describe(
        "Stride values of the slice, a stride can be negative, which causes a reverse slice.");
    TVM_ATTR_FIELD(slice_mode)
        .set_default("end")
        .describe(
            "The slice mode [end, size]."
            "end - The default slice mode, ending indices for the slice."
            "size - The input strides will be ignored, input end in this mode indicates the size"
            "of a slice starting at the location specified by begin. If end[i] is -1,"
            "all remaining elements in that dimension are included in the slice");
  }
};

struct DimAttrs : public tvm::AttrsNode<DimAttrs> {
  Array<Integer> dims;

  TVM_DECLARE_ATTRS(DimAttrs, "relay.attrs.DimAttrs") {
    TVM_ATTR_FIELD(dims).set_default(Array<Integer>()).describe("The dimension sizes.");
  }
};

/*! \brief Attributes for full operator */
struct FullAttrs : public tvm::AttrsNode<FullAttrs> {
  Optional<Array<Integer>> shape;
  DataType dtype;
  double fill_value;

  TVM_DECLARE_ATTRS(FullAttrs, "relay.attrs.FullAttrs") {
    TVM_ATTR_FIELD(shape).describe("Target shape.");
    TVM_ATTR_FIELD(dtype).describe("Target data type.").set_default(NullValue<DataType>());
    TVM_ATTR_FIELD(fill_value).describe("Filled value.");
  }
};

/*! \brief Attributes for swap_axis operator */
struct SwapAxisAttrs : public tvm::AttrsNode<SwapAxisAttrs> {
  int axis1;
  int axis2;
  TVM_DECLARE_ATTRS(SwapAxisAttrs, "attrs.SwapAxisAttrs") {
    TVM_ATTR_FIELD(axis1);
    TVM_ATTR_FIELD(axis2);
  }
};

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
