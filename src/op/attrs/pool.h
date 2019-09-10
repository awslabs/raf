#pragma once

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class MaxPoolAttrs : public ir::AttrsNode<MaxPoolAttrs> {
 public:
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;

  MNM_DECLARE_ATTRS(MaxPoolAttrs, "mnm.attrs.MaxPoolAttrs") {
    MNM_ATTR_FIELD(kernel);    // {h, w}
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
  }
};

class AvgPoolAttrs : public ir::AttrsNode<AvgPoolAttrs> {
 public:
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;
  bool include_pad;

  MNM_DECLARE_ATTRS(AvgPoolAttrs, "mnm.attrs.AvgPoolAttrs") {
    MNM_ATTR_FIELD(kernel);    // {h, w}
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
    MNM_ATTR_FIELD(include_pad);
  }
};

class MaxPoolBackAttrs : public ir::AttrsNode<MaxPoolBackAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;

  MNM_DECLARE_ATTRS(MaxPoolBackAttrs, "mnm.attrs.MaxPoolBackAttrs") {
    MNM_ATTR_FIELD(shape);     // {h, w}
    MNM_ATTR_FIELD(kernel);    // {h, w}
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
  }
};

class AvgPoolBackAttrs : public ir::AttrsNode<AvgPoolBackAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  bool ceil_mode;
  bool include_pad;

  MNM_DECLARE_ATTRS(AvgPoolBackAttrs, "mnm.attrs.AvgPoolBackAttrs") {
    MNM_ATTR_FIELD(shape);     // {h, w}
    MNM_ATTR_FIELD(kernel);    // {h, w}
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
    MNM_ATTR_FIELD(include_pad);
  }
};
}  // namespace attrs
}  // namespace op
}  // namespace mnm
