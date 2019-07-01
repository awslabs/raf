#pragma once

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/value.h>
#include <mnm/enum_base.h>

namespace mnm {
namespace op {
namespace attrs {

enum ActivationMethod {
  Sigmoid,
  Relu,
  TanH,
  ClipRelu,
  Elu,
  Identity
};

class ActivationAttrs : public rly::AttrsNode<ActivationAttrs> {
 public:
  rly::Integer method_;
  rly::Float coef;

  ActivationMethod method() const {
    return (ActivationMethod) ((int) method_);
  }

  MNM_DECLARE_ATTRS(ActivationAttrs, "mnm.attrs.ActivationAttrs") {
    MNM_ATTR_FIELD(method_);
    MNM_ATTR_FIELD(coef);
  }

  static rly::Attrs make(ActivationMethod method, double coef) {
    auto n = rly::make_node<ActivationAttrs>();
    n->method_ = method;
    n->coef = coef;
    return rly::Attrs(n);
  }
};

}  // namespace attr
}  // namespace op
}  // namespace mnm
