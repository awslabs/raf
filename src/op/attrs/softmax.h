#pragma once

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class SoftmaxAttrs : public ir::AttrsNode<SoftmaxAttrs> {
 public:
  mutable int64_t axis;

  MNM_DECLARE_ATTRS(SoftmaxAttrs, "mnm.attrs.SoftmaxAttrs") {
    MNM_ATTR_FIELD(axis);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
