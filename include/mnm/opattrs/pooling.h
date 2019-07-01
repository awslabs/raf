#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

enum PoolingMethod {
  AvgIncludePadding,
  AvgExcludePadding,
  Max
};

class PoolingAttrs : public rly::AttrsNode<PoolingAttrs> {
 public:
  rly::Integer method_;
  rly::Array<rly::Integer> window;
  rly::Array<rly::Integer> padding;
  rly::Array<rly::Integer> stride;

  MNM_DECLARE_ATTRS(PoolingAttrs, "mnm.attrs.PoolingAttrs") {
    MNM_ATTR_FIELD(method_);    // {h, w}
    MNM_ATTR_FIELD(window);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(stride);    // {h, w}
  }

  PoolingMethod method() const {
    return (PoolingMethod) ((int) method_);
  }

  static rly::Attrs make(rly::Integer method,
                         rly::Array<rly::Integer> window,
                         rly::Array<rly::Integer> padding,
                         rly::Array<rly::Integer> stride) {
    auto n = rly::make_node<PoolingAttrs>();
    n->method_ = method;
    n->window = std::move(window);
    n->stride = std::move(stride);
    n->padding = std::move(padding);
    return rly::Attrs(n);
  }
};

}  // namespace attr
}  // namespace op
}  // namespace mnm
