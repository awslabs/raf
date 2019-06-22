#include <mnm/registry.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "./shape_utils.h"

namespace mnm {
namespace value {

using mnm::rly::Array;
using mnm::rly::Integer;
using mnm::rly::make_node;
using mnm::rly::NodePtr;
using mnm::shape_utils::MakeShape;
using mnm::tensor::Tensor;
using mnm::types::Context;
using mnm::types::DType;

TensorValue TensorValue::Assemble(Context ctx,                   //
                                  DType dtype,                   //
                                  std::vector<int64_t> shape,    //
                                  std::vector<int64_t> strides,  //
                                  void* data) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->tensor = Tensor::make(ctx, dtype, shape, strides, data);
  return TensorValue(n);
}

TensorValue AssembleTensorValue(DLContext ctx,            //
                                DLDataType dtype,         //
                                Array<Integer> _shape,    //
                                Array<Integer> _strides,  //
                                void* data) {
  return TensorValue::Assemble(ctx, dtype, MakeShape(_shape), MakeShape(_strides), data);
}

MNM_REGISTER_GLOBAL("mnm.value.AssembleTensorValue").set_body_typed(AssembleTensorValue);

}  // namespace value
}  // namespace mnm
