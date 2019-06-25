#include <mnm/op.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

/*
 * See also
 *   PyTorch: https://pytorch.org/docs/stable/nn.html#relu
 *   TensorFlow: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/relu
 */
namespace mnm {
namespace op {
namespace relu {

using rly::Array;
using rly::Attrs;
using rly::Type;
using rly::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool ReLURel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  reporter->Assign(types[1], types[0]);
  return true;
}

Value ReLUMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  return TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/oshape);
}

// TODO(@were): why clang-format aligns me like that? its inhumane.
MNM_REGISTER_OP("mnm.op.relu")
    .describe(R"code(This is ReLU. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("ReLURel", ReLURel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", ReLUMakeOutput);

}  // namespace relu
}  // namespace op
}  // namespace mnm
