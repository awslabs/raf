/*!
 * Copyright (c) 2019 by Contributors
 * \file transform.h
 * \brief Data structures for transform operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <mnm/ir.h>
#include <array>

namespace mnm {
namespace op {
namespace tvmjit {

using mnm::ir::Array;
using mnm::ir::Integer;
using mnm::ir::Optional;

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

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
