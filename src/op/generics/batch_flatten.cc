#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

/*
 * See also:
 * TF Keras: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/backend/batch_flatten
 *
 * TODO(@junrushao1994): maybe merge this into reshape?
 */
namespace mnm {
namespace op {
namespace batch_flatten {

using ir::Array;
using ir::Attrs;
using ir::TensorTypeNode;
using ir::Type;
using ir::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool BatchFlattenRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  using tvm::Int;
  using tvm::make_const;
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_GE(data->shape.size(), 2);
  auto target_dim = make_const(Int(64), 1);
  // auto target_dim = TVMIntImm::make(DType(DTypeCode::kInt(), 64), (int64_t)1);
  for (int i = 1; i < static_cast<int>(data->shape.size()); ++i) {
    target_dim = target_dim * data->shape[i];
  }
  reporter->Assign(types[1], TensorTypeNode::make({data->shape[0], target_dim}, data->dtype));
  return true;
}

OpInfo BatchFlattenMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  const int64_t* dshape = data->shape;
  const int ndim = data->ndim;
  CHECK_GE(ndim, 2) << "ValueError: batch_flatten only works with ndim >= 2";
  const int64_t nbatch{dshape[0]};
  int64_t flat{1};
  for (int i = 1; i < ndim; ++i) {
    flat = flat * int64_t{dshape[i]};
  }
  return OpInfo::make(
      TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/{nbatch, flat}),
      data->ctx);
}

MNM_REGISTER_OP("mnm.op.batch_flatten")
    .describe(R"code(This is BatchFlatten. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "4D Tensor", "Input data.")
    .add_type_rel("BatchFlattenRel", BatchFlattenRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", BatchFlattenMakeOutput);

}  // namespace batch_flatten
}  // namespace op
}  // namespace mnm
