#include <mnm/op.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/types.h>
#include <mnm/value.h>

#include "../shape_utils.h"

/*
 * See also:
 * TF Keras: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/backend/batch_flatten
 *
 * TODO(@junrushao1994): maybe merge this into reshape?
 *
 * This is essentially transposed matrix multiplication.
 * [..., a] * [b, a] => [..., b]
 */
namespace mnm {
namespace op {
namespace linear {

using mnm::rly::Array;
using mnm::rly::Attrs;
using mnm::rly::IndexExpr;
using mnm::rly::TensorTypeNode;
using mnm::rly::Type;
using mnm::rly::TypeReporter;
using mnm::shape_utils::MakeShape;
using mnm::tensor::Tensor;
using mnm::value::TensorValue;
using mnm::value::Value;

bool LinearRel(const Array<Type>& types,  //
               int num_inputs,            //
               const Attrs& attrs,        //
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK(weight != nullptr);
  const Array<IndexExpr>& dshape = data->shape;
  const Array<IndexExpr>& wshape = weight->shape;
  CHECK_GE(dshape.size(), 1);
  CHECK_EQ(wshape.size(), 2);
  int ndim = dshape.size();
  const IndexExpr& in_units = dshape[ndim - 1];
  const IndexExpr& out_units = wshape[0];
  const IndexExpr& _in_units = wshape[1];
  reporter->AssertEQ(in_units, _in_units);
  Array<IndexExpr> oshape = dshape;
  oshape.Set(ndim - 1, out_units);
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Value LinearMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  const Tensor& data = values[0];
  const Tensor& weight = values[1];
  CHECK_GE(data->ndim, 1);
  CHECK_EQ(weight->ndim, 2);
  int ndim = data->ndim;
  int64_t in_units = data->shape[ndim - 1];
  int64_t out_units = weight->shape[0];
  int64_t _in_units = weight->shape[1];
  CHECK_EQ(in_units, _in_units);
  std::vector<int64_t> oshape(data->shape, data->shape + ndim);
  oshape[ndim - 1] = out_units;
  return TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype, /*shape=*/oshape);
}

// TODO(@were): why clang-format aligns me like that? its inhumane.
MNM_REGISTER_OP("mnm.op.linear")
    .describe(R"code(This is Linear. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", ">=1D Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Filter.")
    .add_type_rel("LinearRel", LinearRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", LinearMakeOutput);

}  // namespace linear
}  // namespace op
}  // namespace mnm
