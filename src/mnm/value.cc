#include <mnm/registry.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "./common/shape_utils.h"

namespace mnm {
namespace value {

using common::shape_utils::MakeShape;
using rly::Array;
using rly::Integer;
using rly::make_node;
using rly::NodePtr;
using tensor::Tensor;

Value::operator const DLTensor*() const {
  if (auto tensor_value = this->as<TensorValueNode>()) {
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

TensorValue TensorValue::Assemble(Context ctx,                   //
                                  DType dtype,                   //
                                  std::vector<int64_t> shape,    //
                                  std::vector<int64_t> strides,  //
                                  void* data) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->tensor = Tensor::make(ctx, dtype, shape, strides, data);
  return TensorValue(n);
}

TensorValue AssembleTensorValue(DLContext ctx,           //
                                DLDataType dtype,        //
                                Array<Integer> shape,    //
                                Array<Integer> strides,  //
                                void* data) {
  return TensorValue::Assemble(ctx, dtype, MakeShape(shape), MakeShape(strides), data);
}

MNM_REGISTER_GLOBAL("mnm.value.AssembleTensorValue").set_body_typed(AssembleTensorValue);

}  // namespace value
}  // namespace mnm
