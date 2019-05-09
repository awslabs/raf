#pragma once

#include <mnm/rly.h>
#include <mnm/tensor.h>

namespace mnm {
namespace ndarray {

class NDArray;

class NDArrayNode final : public mnm::rly::Node {
 public:
  mnm::tensor::Tensor data{nullptr};
  mnm::rly::Expr expr{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("_data", &data);
    v->Visit("expr", &expr);
  }
  static constexpr const char* _type_key = "mnm.ndarray";
  MNM_DEF_NODE_TYPE_INFO(NDArrayNode, mnm::rly::Node);

 public:
  struct Impl;
};

MNM_DEF_NODE_REF(NDArray, NDArrayNode, mnm::rly::NodeRef);

}  // namespace ndarray
}  // namespace mnm
