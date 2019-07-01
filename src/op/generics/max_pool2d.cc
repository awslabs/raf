#include <mnm/op.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../attrs/pooling.h"

/*
 * See also:
 *   PyTorch: https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
 *   TensorFlow: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/max_pool2d
 *
 * TODO(@junrushao1994): NCHW + OIHW for now, more layouts tbd
 */

namespace mnm {
namespace op {
namespace max_pool2d {

using attrs::MaxPool2DAttrs;
using rly::Array;
using rly::Attrs;
using rly::IndexExpr;
using rly::TensorTypeNode;
using rly::Type;
using rly::TypeReporter;
using tensor::Tensor;
using value::TensorValue;
using value::Value;

bool MaxPool2DRel(const Array<Type>& types,  //
                  int num_inputs,            //
                  const Attrs& attrs,        //
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* param = attrs.as<attrs::MaxPool2DAttrs>();
  CHECK(data != nullptr);
  CHECK(param != nullptr);
  CHECK_EQ(data->shape.size(), 4);
  CHECK_EQ(param->kernel_size.size(), 2);
  CHECK_EQ(param->stride.size(), 2);
  CHECK_EQ(param->padding.size(), 2);
  CHECK_EQ(param->dilation.size(), 2);
  const IndexExpr& n_in = data->shape[0];
  const IndexExpr& c_in = data->shape[1];
  const IndexExpr& h_in = data->shape[2];
  const IndexExpr& w_in = data->shape[3];
  const IndexExpr& kernel_h = param->kernel_size[0];
  const IndexExpr& kernel_w = param->kernel_size[1];
  const IndexExpr& stride_h = param->stride[0];
  const IndexExpr& stride_w = param->stride[1];
  const IndexExpr& pad_h = param->padding[0];
  const IndexExpr& pad_w = param->padding[1];
  const IndexExpr& dilate_h = param->dilation[0];
  const IndexExpr& dilate_w = param->dilation[1];
  IndexExpr h_out, w_out;
  if (!param->ceil_mode) {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  reporter->Assign(types[1], TensorTypeNode::make({n_in, c_in, h_out, w_out}, data->dtype));
  return true;
}

Value MaxPool2DMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 1);
  const Tensor& data = values[0];
  const auto* param = attrs.as<attrs::MaxPool2DAttrs>();
  CHECK(param != nullptr);
  CHECK_EQ(data->ndim, 4);
  CHECK_EQ(param->kernel_size.size(), 2);
  CHECK_EQ(param->stride.size(), 2);
  CHECK_EQ(param->padding.size(), 2);
  CHECK_EQ(param->dilation.size(), 2);
  int64_t n_in = data->shape[0];
  int64_t c_in = data->shape[1];
  int64_t h_in = data->shape[2];
  int64_t w_in = data->shape[3];
  int64_t kernel_h = param->kernel_size[0]->value;
  int64_t kernel_w = param->kernel_size[1]->value;
  int64_t stride_h = param->stride[0]->value;
  int64_t stride_w = param->stride[1]->value;
  int64_t pad_h = param->padding[0]->value;
  int64_t pad_w = param->padding[1]->value;
  int64_t dilate_h = param->dilation[0]->value;
  int64_t dilate_w = param->dilation[1]->value;
  int64_t h_out, w_out;
  if (!param->ceil_mode) {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  return TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype,
                               /*shape=*/{n_in, c_in, h_out, w_out});
}

// TODO(@were): why clang-format aligns me like that? its inhumane.
MNM_REGISTER_OP("mnm.op.max_pool2d")
    .describe(R"code(This is MaxPool2D. Have a nice day.
)code" MNM_ADD_FILELINE)
    .set_attrs_type_key("mnm.attrs.MaxPool2DAttrs")
    .set_num_inputs(2)
    .add_argument("data", "4D Tensor", "Input data.")
    .add_type_rel("MaxPool2DRel", MaxPool2DRel)
    .set_attr<FOpMakeOutput>("FOpMakeOutput", MaxPool2DMakeOutput);

}  // namespace max_pool2d
}  // namespace op
}  // namespace mnm
