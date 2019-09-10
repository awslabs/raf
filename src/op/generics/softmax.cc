#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../attrs/softmax.h"

namespace mnm {
namespace op {
namespace unary {

using ir::Array;
using ir::Attrs;
using ir::TensorTypeNode;
using ir::Type;
using ir::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool SoftmaxRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), num_inputs);
  auto* ref = types[0].as<TensorTypeNode>();
  for (int i = 1; i < num_inputs; ++i) {
    auto* cur = types[i].as<TensorTypeNode>();
    CHECK_EQ(cur->shape.size(), ref->shape.size());
    for (int j = 0; j < (int)cur->shape.size(); ++j) {
      reporter->AssertEQ(cur->shape[j], ref->shape[j]);
    }
  }
  reporter->Assign(types[num_inputs - 1], types[0]);
  return true;
}

template <int NInputs>
OpInfo SoftmaxMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), NInputs);
  const Tensor& data = values[0];
  auto sm_attrs = attrs.as<attrs::SoftmaxAttrs>();
  if (sm_attrs->axis < 0) {
    sm_attrs->axis += data->ndim;
  }
  CHECK(sm_attrs->axis >= 0 && sm_attrs->axis < data->ndim);
  for (int i = 1; i < NInputs; ++i) {
    const Tensor& cur = values[i];
    CHECK_EQ(cur->ndim, data->ndim);
    for (int j = 0; j < cur->ndim; ++j) {
      CHECK_EQ(cur->shape[j], data->shape[j]);
    }
  }
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  return OpInfo::make(
      TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/oshape), data->ctx);
}

MNM_REGISTER_OP("mnm.op.softmax")
    .describe(R"code(This is softmax.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type_key("mnm.attrs.SoftmaxAttrs")
    .add_type_rel("SoftmaxRel", SoftmaxRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", SoftmaxMakeOutput<1>);

MNM_REGISTER_OP("mnm.op.log_softmax")
    .describe(R"code(This is softmax.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type_key("mnm.attrs.SoftmaxAttrs")
    .add_type_rel("SoftmaxRel", SoftmaxRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", SoftmaxMakeOutput<1>);

MNM_REGISTER_OP("mnm.op.grad.softmax")
    .describe(R"code(This is backward softmax.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type_key("mnm.attrs.SoftmaxAttrs")
    .add_type_rel("SoftmaxBackRel", SoftmaxRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", SoftmaxMakeOutput<2>);

}  // namespace unary
}  // namespace op
}  // namespace mnm
