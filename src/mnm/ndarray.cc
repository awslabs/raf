#include <mnm/ndarray.h>
#include <mnm/registry.h>

namespace mnm {
namespace ndarray {

using mnm::ndarray::NDArray;
using mnm::registry::Registry;
using mnm::rly::Expr;
using mnm::rly::make_node;
using mnm::rly::NodePtr;
using mnm::tensor::Tensor;

NDArray NDArrayNode::make(Tensor data, Expr expr) {
  NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
  n->data = std::move(data);
  n->expr = std::move(expr);
  return NDArray(n);
}

NDArray NDArrayNode::power(NDArray x1, NDArray x2) {
  static const auto* op = Registry::Get("relay.op._make.power");
  NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
  n->expr = (*op)(x1, x2);
  return NDArray(n);
}

MNM_REGISTER_GLOBAL("mnm._make.NDArray").set_body_typed(NDArrayNode::make);

}  // namespace ndarray
}  // namespace mnm
