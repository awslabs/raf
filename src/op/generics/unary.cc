#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>
#include <topi/elemwise.h>

/*
 * See also
 *   PyTorch: https://pytorch.org/docs/stable/nn.html#relu
 *   TensorFlow: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/relu
 */
namespace mnm {
namespace op {
namespace unary {

using ir::Array;
using ir::Op;
using ir::Attrs;
using ir::FTVMCompute;
using ir::FTVMSchedule;
using ir::TensorTypeNode;
using ir::TOpPattern;
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

OpInfo IdenticalMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  return OpInfo::make(
      TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/oshape), data->ctx);
}

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

MNM_REGISTER_OP("mnm.op.copy")
    .set_num_inputs(1)
    .add_argument("a", "array_like", "Input data.")
    .add_type_rel("IdenticalRel", IdenticalRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalMakeOutput)
    .set_attr<TOpPattern>("TOpPattern", tvm::relay::kElemWise)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<tvm::Tensor>& inputs,
                              const Type& out_type,
                              const tvm::Target& target) -> Array<tvm::Tensor> {
                             return {topi::identity(inputs[0])};
                           })
    .set_attr<FTVMSchedule>("FTVMSchedule",
                            [](const Attrs& attrs, const Array<tvm::Tensor>& outs,
                               const tvm::Target& target) -> tvm::Schedule {
                              static auto fschedule = Op::GetAttr<FTVMSchedule>("FTVMSchedule")[Op::Get("copy")];
                              return fschedule(attrs, outs, target);
                            });

bool ActivationBackRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  int n = types.size();
  CHECK_EQ(n, num_inputs + 1);  // y, dy, x, dx
  const auto* out = types[0].as<TensorTypeNode>();
  CHECK(out != nullptr);
  for (int i = 1; i < n; ++i) {
    const auto* arg = types[i].as<TensorTypeNode>();
    CHECK_EQ(arg->shape.size(), out->shape.size());
    int m = arg->shape.size();
    for (int j = 0; j < m; ++j) {
      reporter->AssertEQ(arg->shape[j], out->shape[j]);
    }
  }
  reporter->Assign(types[3], types[0]);
  return true;
}

template <int NInputs>
OpInfo IdenticalBackMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), NInputs);  // y, dy, x
  const Tensor& y = values[0];
  for (int i = 1; i < NInputs; ++i) {
    const Tensor& arg = values[i];
    CHECK_EQ(arg->ndim, y->ndim);
    int m = arg->ndim;
    for (int j = 0; j < m; ++j) {
      CHECK_EQ(arg->shape[j], y->shape[j]);
    }
  }
  std::vector<int64_t> oshape(y->shape, y->shape + y->ndim);
  return OpInfo::make(TensorValue::Assemble(/*ctx=*/y->ctx, /*dtype=*/y->dtype, /*shape=*/oshape),
                      y->ctx);
}

MNM_REGISTER_OP("mnm.op.grad.relu")
    .describe(R"code(This backward relu.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_type_rel("ReLUBackRel", ActivationBackRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalBackMakeOutput<3>);

MNM_REGISTER_OP("mnm.op.grad.tanh")
    .describe(R"code(This is backward tanh.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_type_rel("TanHBackRel", ActivationBackRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalBackMakeOutput<3>);

MNM_REGISTER_OP("mnm.op.grad.sigmoid")
    .describe(R"code(This is backward sigmoid.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_type_rel("SigmoidBackRel", ActivationBackRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", IdenticalBackMakeOutput<3>);

}  // namespace unary
}  // namespace op
}  // namespace mnm
