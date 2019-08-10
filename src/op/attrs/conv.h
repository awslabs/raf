#pragma once

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class Conv2DAttrs : public ir::AttrsNode<Conv2DAttrs> {
 public:
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> padding;
  ir::Array<ir::Integer> dilation;
  ir::Integer groups;

  MNM_DECLARE_ATTRS(Conv2DAttrs, "mnm.attrs.Conv2DAttrs") {
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(groups);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
