#pragma once

#include <mnm/rly.h>
#include <mnm/tensor.h>

namespace mnm {
namespace ndarray {

class NDArray;

class NDArrayNode : public mnm::rly::Node {
 public:
  mnm::tensor::Tensor data{nullptr};
  mnm::rly::Expr expr{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("expr", &expr);
  }

  static mnm::ndarray::NDArray make(mnm::tensor::Tensor data, mnm::rly::Expr expr);

  static mnm::ndarray::NDArray power(mnm::ndarray::NDArray x1, mnm::ndarray::NDArray x2);

  static constexpr const char* _type_key = "mnm.NDArray";
  MNM_DEF_NODE_TYPE_INFO(NDArrayNode, mnm::rly::Node);
};

MNM_DEF_NODE_REF(NDArray, NDArrayNode, mnm::rly::NodeRef);

}  // namespace ndarray
}  // namespace mnm
