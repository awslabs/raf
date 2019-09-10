#pragma once

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/value.h>

namespace mnm {
namespace op {
namespace attrs {

class LocalResponseNormAttrs : public ir::AttrsNode<LocalResponseNormAttrs> {
 public:
  int64_t n;
  double alpha;
  double beta;
  double k;

  MNM_DECLARE_ATTRS(LocalResponseNormAttrs, "mnm.attrs.LocalResponseNormAttrs") {
    MNM_ATTR_FIELD(n);
    MNM_ATTR_FIELD(alpha);
    MNM_ATTR_FIELD(beta);
    MNM_ATTR_FIELD(k);
  }
};

class BatchNormAttrs : public ir::AttrsNode<BatchNormAttrs> {
 public:
  double eps;
  double momentum;
  mutable bool is_training;

  MNM_DECLARE_ATTRS(BatchNormAttrs, "mnm.attrs.BatchNormAttrs") {
    MNM_ATTR_FIELD(eps);
    MNM_ATTR_FIELD(momentum);
    MNM_ATTR_FIELD(is_training);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace mnm
