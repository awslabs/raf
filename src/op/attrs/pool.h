#pragma once

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class MaxPoolAttrs : public ir::AttrsNode<MaxPoolAttrs> {
 public:
  ir::Array<ir::Integer> kernel_size;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;

  MNM_DECLARE_ATTRS(MaxPoolAttrs, "mnm.attrs.MaxPoolAttrs") {
    MNM_ATTR_FIELD(kernel_size);  // {h, w}
    MNM_ATTR_FIELD(stride);       // {h, w}
    MNM_ATTR_FIELD(padding);      // {h, w}
    MNM_ATTR_FIELD(dilation);     // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
  }
};

class AvgPoolAttrs : public ir::AttrsNode<AvgPoolAttrs> {
 public:
  ir::Array<ir::Integer> kernel_size;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;
  bool include_pad;

  MNM_DECLARE_ATTRS(AvgPoolAttrs, "mnm.attrs.AvgPoolAttrs") {
    MNM_ATTR_FIELD(kernel_size);  // {h, w}
    MNM_ATTR_FIELD(stride);       // {h, w}
    MNM_ATTR_FIELD(padding);      // {h, w}
    MNM_ATTR_FIELD(dilation);     // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
    MNM_ATTR_FIELD(include_pad);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
