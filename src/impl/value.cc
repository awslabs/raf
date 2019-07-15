#include <mnm/registry.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../common/shape_utils.h"

namespace mnm {
namespace value {

using common::shape_utils::MakeShape;
using rly::Array;
using rly::Function;
using rly::Integer;
using rly::make_node;
using rly::Map;
using rly::NodePtr;
using rly::Var;
using tensor::Tensor;

TensorValue TensorValue::make(tensor::Tensor tensor) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->tensor = std::move(tensor);
  return TensorValue(n);
}

TupleValue TupleValue::make(Array<Value> fields) {
  NodePtr<TupleValueNode> n = make_node<TupleValueNode>();
  n->fields = std::move(fields);
  return TupleValue(n);
}

ClosureValue ClosureValue::make(Map<Var, Value> env, Function func) {
  NodePtr<ClosureValueNode> n = make_node<ClosureValueNode>();
  n->env = std::move(env);
  n->func = std::move(func);
  return ClosureValue(n);
}

RefValue RefValue::make(Value value) {
  NodePtr<RefValueNode> n = make_node<RefValueNode>();
  n->value = std::move(value);
  return RefValue(n);
}

Value::operator const DLTensor*() const {
  if (const auto* tensor_value = this->as<TensorValueNode>()) {
    const DLTensor* dl_tensor_ref = tensor_value->tensor.operator->();
    return dl_tensor_ref;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

Value::operator const tensor::Tensor&() const {
  if (const auto* tensor_value = this->as<TensorValueNode>()) {
    return tensor_value->tensor;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

TensorValue TensorValue::Assemble(const Context& ctx, const DType& dtype,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& strides, void* const data) {
  return TensorValue::make(Tensor::make(ctx, dtype, shape, strides, data));
}

TensorValue AssembleTensorValue(DLContext ctx, DLDataType dtype, Array<Integer> shape,
                                Array<Integer> strides, void* data) {
  return TensorValue::make(Tensor::make(ctx, dtype, MakeShape(shape), MakeShape(strides), data));
}

MNM_REGISTER_GLOBAL("mnm.value.AssembleTensorValue").set_body_typed(AssembleTensorValue);

}  // namespace value
}  // namespace mnm
