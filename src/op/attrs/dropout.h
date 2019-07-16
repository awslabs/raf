#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class DropoutAttrs : public rly::AttrsNode<DropoutAttrs> {
 public:
  rly::Float dropout;

  MNM_DECLARE_ATTRS(DropoutAttrs, "mnm.attrs.DropoutAttrs") {
    MNM_ATTR_FIELD(dropout);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
