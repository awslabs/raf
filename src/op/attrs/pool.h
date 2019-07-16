#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class MaxPoolAttrs : public rly::AttrsNode<MaxPoolAttrs> {
 public:
  rly::Array<rly::Integer> kernel_size;
  rly::Array<rly::Integer> stride;
  rly::Array<rly::Integer> padding;
  rly::Array<rly::Integer> dilation;
  bool ceil_mode;

  MNM_DECLARE_ATTRS(MaxPoolAttrs, "mnm.attrs.MaxPoolAttrs") {
    MNM_ATTR_FIELD(kernel_size);  // {h, w}
    MNM_ATTR_FIELD(stride);       // {h, w}
    MNM_ATTR_FIELD(padding);      // {h, w}
    MNM_ATTR_FIELD(dilation);     // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
  }
};

class AvgPoolAttrs : public rly::AttrsNode<AvgPoolAttrs> {
 public:
  rly::Array<rly::Integer> kernel_size;
  rly::Array<rly::Integer> stride;
  rly::Array<rly::Integer> padding;
  rly::Array<rly::Integer> dilation;
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
