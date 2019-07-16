#include <mnm/op.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include <tvm/node/memory.h>

// TODO(@were): Implement dropout!

namespace mnm {
namespace op {
namespace dropout {

using rly::Array;
using rly::Attrs;
using rly::DataType;
using rly::TensorTypeNode;
using rly::TupleType;
using rly::Type;
using rly::TypeReporter;
using tensor::Tensor;
using tvm::make_node;
using value::OpaqueValue;
using value::OpaqueValueNode;
using value::TensorValue;
using value::TupleValue;
using value::Value;

bool DropoutRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  // TODO(@were): For now, we use 0-dom tensor as the type placeholder of the opaque values.
  Type empty_tensor = TensorTypeNode::make({}, DataType());
  TupleType res = rly::TupleTypeNode::make({types[0], empty_tensor});
  reporter->Assign(types[1], res);
  return true;
}

Value DropoutMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  TensorValue out = TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype,
                                          /*shape=*/oshape);
  OpaqueValue states(make_node<OpaqueValueNode>());
  return TupleValue::make({out, states});
}

MNM_REGISTER_OP("mnm.op.dropout")
    .describe(R"code(This is dropout.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("DropoutdRel", DropoutRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", DropoutMakeOutput);

}  // namespace dropout
}  // namespace op
}  // namespace mnm
