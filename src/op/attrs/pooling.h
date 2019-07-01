#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class MaxPool2DAttrs : public rly::AttrsNode<MaxPool2DAttrs> {
 public:
  rly::Array<rly::Integer> kernel_size;
  rly::Array<rly::Integer> stride;
  rly::Array<rly::Integer> padding;
  rly::Array<rly::Integer> dilation;
  bool ceil_mode;

  MNM_DECLARE_ATTRS(MaxPool2DAttrs, "mnm.attrs.MaxPool2DAttrs") {
    MNM_ATTR_FIELD(kernel_size);  // {h, w}
    MNM_ATTR_FIELD(stride);       // {h, w}
    MNM_ATTR_FIELD(padding);      // {h, w}
    MNM_ATTR_FIELD(dilation);     // {h, w}
    MNM_ATTR_FIELD(ceil_mode);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
