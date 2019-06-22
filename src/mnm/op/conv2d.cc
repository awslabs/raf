#include <mnm/op.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>
#include <mnm/types.h>
#include <mnm/value.h>

#include "../shape_utils.h"

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

using mnm::rly::Array;
using mnm::rly::Attrs;
using mnm::rly::AttrsNode;
using mnm::rly::IndexExpr;
using mnm::rly::Integer;
using mnm::rly::make_node;
using mnm::rly::TensorTypeNode;
using mnm::rly::Type;
using mnm::rly::TypeReporter;
using mnm::shape_utils::MakeShape;
using mnm::tensor::Tensor;
using mnm::value::TensorValue;
using mnm::value::Value;

class Conv2DAttrs : public AttrsNode<Conv2DAttrs> {
 public:
  Array<Integer> stride;
  Array<Integer> padding;
  Array<Integer> dilation;
  Integer groups;

  MNM_DECLARE_ATTRS(Conv2DAttrs, "mnm.attrs.Conv2DAttrs") {
    MNM_ATTR_FIELD(stride);    // {h, w}
    MNM_ATTR_FIELD(padding);   // {h, w}
    MNM_ATTR_FIELD(dilation);  // {h, w}
    MNM_ATTR_FIELD(groups);
  }

  static Attrs make(Array<Integer> stride,    //
                    Array<Integer> padding,   //
                    Array<Integer> dilation,  //
                    Integer groups) {
    auto n = make_node<Conv2DAttrs>();
    n->stride = std::move(stride);
    n->padding = std::move(padding);
    n->dilation = std::move(dilation);
    n->groups = std::move(groups);
    return Attrs(n);
  }
};

bool Conv2DRel(const Array<Type>& types,  //
               int num_inputs,            //
               const Attrs& attrs,        //
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* param = attrs.as<Conv2DAttrs>();
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

Value Conv2dMakeOutput(const Array<Value>& values, const Attrs& attrs) {
  CHECK_EQ(values.size(), 2);
  const Tensor& data = values[0];
  const Tensor& weight = values[1];
  const auto* param = attrs.as<Conv2DAttrs>();
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
  return TensorValue::Assemble(/*ctx=*/data->ctx, /*dtype=*/data->dtype,
                               /*shape=*/MakeShape({n_in, out, h_out, w_out}));
}

MNM_REGISTER_NODE_TYPE(Conv2DAttrs);
MNM_REGISTER_GLOBAL("mnm.attrs._make.Conv2DAttrs").set_body_typed(Conv2DAttrs::make);

// TODO(@were): why clang-format aligns me like that? its inhumane.
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
