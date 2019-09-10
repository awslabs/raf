#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../../common/shape_utils.h"

/*
 * See also:
 * PyTorch: https://pytorch.org/docs/stable/nn.html#batchnorm2d
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

bool BatchNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  using tvm::Int;
  using tvm::make_const;
  CHECK_EQ(types.size(), num_inputs + 1);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_GE(data->shape.size(), 2);
  for (int i = 1; i < num_inputs; ++i) {
    auto* param = types[i].as<TensorTypeNode>();
    CHECK(param != nullptr);
    CHECK_EQ(param->shape.size(), 1);
    reporter->AssertEQ(param->shape[0], data->shape[1]);
  }
  reporter->Assign(types[num_inputs], TensorTypeNode::make(data->shape, data->dtype));
  return true;
}

template <int NDims>
OpInfo BatchNormMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 5);
  const Tensor& data = values[0];
  const int64_t* dshape = data->shape;
  const int ndim = data->ndim;
  CHECK_GE(ndim, NDims + 2) << "ValueError: batch_flatten only works with ndim >= 2";
  for (int i = 1; i < (int)values.size(); ++i) {
    const Tensor& tensor = values[i];
    CHECK_EQ(tensor->ndim, 1);
    CHECK_EQ(tensor->shape[0], data->shape[1]);
  }
  std::vector<int64_t> oshape(dshape, dshape + ndim);
  return OpInfo::make(TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype,
                                            /*shape=*/oshape),
                      data->ctx);
}

MNM_REGISTER_OP("mnm.op.batch_norm2d")
    .describe(R"code(This is BatchFlatten. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_attrs_type_key("mnm.attrs.BatchNormAttrs")
    .set_num_inputs(5)
    .add_argument("data", "4D Tensor", "Input data.")
    .add_type_rel("BatchFlattenRel", BatchNormRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", BatchNormMakeOutput<2>);

}  // namespace batch_flatten
}  // namespace op
}  // namespace mnm
