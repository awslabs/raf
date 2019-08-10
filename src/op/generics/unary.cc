#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

/*
 * See also
 *   PyTorch: https://pytorch.org/docs/stable/nn.html#relu
 *   TensorFlow: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/relu
 */
namespace mnm {
namespace op {
namespace identical {

using ir::Array;
using ir::Attrs;
using ir::Type;
using ir::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool IdenticalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  reporter->Assign(types[1], types[0]);
  return true;
}

Value IdenticalMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  return TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/oshape);
}

MNM_REGISTER_OP("mnm.op.add")
    .describe(R"code(This is Add.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("AddRel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput);

MNM_REGISTER_OP("mnm.op.relu")
    .describe(R"code(Apply a relu elmentwisely on the given tensor.

    This op creates a relu layer.

    - **data**: A any-dimension tensor.
    - **out**: The output tensor. The tensor on which relu applied.

)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("ReLURel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput);

MNM_REGISTER_OP("mnm.op.tanh")
    .describe(R"code(This is TanH. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("TanHRel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput);

MNM_REGISTER_OP("mnm.op.sigmoid")
    .describe(R"code(This is Sigmoid. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("SigmoidRel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput);

MNM_REGISTER_OP("mnm.op.softmax")
    .describe(R"code(This is softmax.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Any Tensor", "Input data.")
    .add_type_rel("SoftmaxRel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput);

}  // namespace identical
}  // namespace op
}  // namespace mnm
