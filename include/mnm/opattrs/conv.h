#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class Conv2DAttrs : public rly::AttrsNode<Conv2DAttrs> {
 public:
  rly::Array<rly::Integer> stride;
  rly::Array<rly::Integer> padding;
  rly::Array<rly::Integer> dilation;
  rly::Integer groups;

  MNM_DECLARE_ATTRS(Conv2DAttrs, "mnm.attrs.Conv2DAttrs") {
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(groups);
  }

  static rly::Attrs make(rly::Array<rly::Integer> stride,    //
                    rly::Array<rly::Integer> padding,   //
                    rly::Array<rly::Integer> dilation,  //
                    rly::Integer groups) {
    auto n = rly::make_node<Conv2DAttrs>();
    n->stride = std::move(stride);
    n->padding = std::move(padding);
    n->dilation = std::move(dilation);
    n->groups = std::move(groups);
    return rly::Attrs(n);
  }
};

}  // namespace attr
}  // namespace op
}  // namespace mnm