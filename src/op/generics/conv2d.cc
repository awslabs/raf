#include <mnm/ir.h>
#include <mnm/op.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../attrs/conv.h"

/*
 * See also:
 *   PyTorch: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
 *   TensorFlow: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/conv2d
 *
 * TODO(@junrushao1994): NCHW + OIHW for now, more layouts tbd
 */

namespace mnm {
namespace op {
namespace conv2d {

using attrs::Conv2DAttrs;
using ir::Array;
using ir::Attrs;
using ir::IndexExpr;
using ir::TensorTypeNode;
using ir::Type;
using ir::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool Conv2DRel(const Array<Type>& types,  //
               int num_inputs,            //
               const Attrs& attrs,        //
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* param = attrs.as<attrs::Conv2DAttrs>();
  CHECK(data != nullptr);
  CHECK(weight != nullptr);
  CHECK(param != nullptr);
  CHECK_EQ(data->shape.size(), 4);
  CHECK_EQ(weight->shape.size(), 4);
  CHECK_EQ(param->stride.size(), 2);
  CHECK_EQ(param->padding.size(), 2);
  CHECK_EQ(param->dilation.size(), 2);
  // TODO(@junrushao1994): `groups` is not used
  const IndexExpr& n_in = data->shape[0];
  const IndexExpr& c_in = data->shape[1];
  const IndexExpr& h_in = data->shape[2];
  const IndexExpr& w_in = data->shape[3];
  const IndexExpr& out = weight->shape[0];
  const IndexExpr& in = weight->shape[1];
  const IndexExpr& kernel_h = weight->shape[2];
  const IndexExpr& kernel_w = weight->shape[3];
  const IndexExpr& stride_h = param->stride[0];
  const IndexExpr& stride_w = param->stride[1];
  const IndexExpr& pad_h = param->padding[0];
  const IndexExpr& pad_w = param->padding[1];
  const IndexExpr& dilate_h = param->dilation[0];
  const IndexExpr& dilate_w = param->dilation[1];
  IndexExpr h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  IndexExpr w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  reporter->AssertEQ(c_in, in);
  reporter->Assign(types[2], TensorTypeNode::make({n_in, out, h_out, w_out}, data->dtype));
  return true;
}

OpInfo Conv2dMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  const Tensor& data = values[0];
  const Tensor& weight = values[1];
  const auto* param = attrs.as<attrs::Conv2DAttrs>();
  CHECK(param != nullptr);
  CHECK_EQ(data->ndim, 4);
  CHECK_EQ(weight->ndim, 4);
  CHECK_EQ(param->stride.size(), 2);
  CHECK_EQ(param->padding.size(), 2);
  CHECK_EQ(param->dilation.size(), 2);
  int64_t n_in = data->shape[0];
  int64_t c_in = data->shape[1];
  int64_t h_in = data->shape[2];
  int64_t w_in = data->shape[3];
  int64_t out = weight->shape[0];
  int64_t in = weight->shape[1];
  int64_t kernel_h = weight->shape[2];
  int64_t kernel_w = weight->shape[3];
  int64_t stride_h = param->stride[0]->value;
  int64_t stride_w = param->stride[1]->value;
  int64_t pad_h = param->padding[0]->value;
  int64_t pad_w = param->padding[1]->value;
  int64_t dilate_h = param->dilation[0]->value;
  int64_t dilate_w = param->dilation[1]->value;
  int64_t h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  int64_t w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  CHECK_EQ(c_in, in);
  return OpInfo::make(TensorValue::Assemble(/*ctx=*/data->ctx,
                                            /*dtype=*/data->dtype,
                                            /*shape=*/{n_in, out, h_out, w_out}),
                      data->ctx);
}

MNM_REGISTER_OP("mnm.op.conv2d")
    .describe(R"code(This is Conv2d. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_attrs_type_key("mnm.attrs.Conv2dAttrs")
    .set_num_inputs(2)
    .add_argument("data", "4D Tensor", "Input data.")
    .add_argument("weight", "4D Tensor", "Filter.")
    .add_type_rel("Conv2dRel", Conv2DRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", Conv2dMakeOutput);

}  // namespace conv2d
}  // namespace op
}  // namespace mnm
