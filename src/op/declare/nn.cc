/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include <tvm/tir/data_layout.h>
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/tensor.h"
#include "../schema/nn.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::ir;
using namespace mnm::value;

void Conv2D(const CallValues& call) {
  // N.B.: NCHW + OIHW
  const auto* args = call->args.as<ConvArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* w = args->w;
  CHECK_EQ(x->ndim, 4);
  CHECK_EQ(w->ndim, 4);
  // TODO(@junrushao1994): deduce ctx here
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);

  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape{
      tvm::Integer(x->shape[0]),
      tvm::Integer(x->shape[1]),
      tvm::Integer(x->shape[2]),
      tvm::Integer(x->shape[3]),
  };
  tvm::tir::BijectiveLayout w_layout_converter(args->kernel_layout, "OIHW");
  tvm::Array<tvm::PrimExpr> w_shape{
      tvm::Integer(w->shape[0]),
      tvm::Integer(w->shape[1]),
      tvm::Integer(w->shape[2]),
      tvm::Integer(w->shape[3]),
  };

  in_shape = data_layout_converter.ForwardShape(in_shape);
  w_shape = w_layout_converter.ForwardShape(w_shape);

  int64_t n_in = in_shape[0].as<tvm::IntImmNode>()->value;
  int64_t c_in = in_shape[1].as<tvm::IntImmNode>()->value;
  int64_t h_in = in_shape[2].as<tvm::IntImmNode>()->value;
  int64_t w_in = in_shape[3].as<tvm::IntImmNode>()->value;
  int64_t out = w_shape[0].as<tvm::IntImmNode>()->value;
  int64_t in = w_shape[1].as<tvm::IntImmNode>()->value;
  int64_t kernel_h = w_shape[2].as<tvm::IntImmNode>()->value;
  int64_t kernel_w = w_shape[3].as<tvm::IntImmNode>()->value;
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];

  int64_t pad_h;
  int64_t pad_w;
  GetPadHW(args->padding, &pad_h, &pad_w);

  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out = (h_in + pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  int64_t w_out = (w_in + pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  int64_t groups = args->groups;
  CHECK_EQ(c_in / groups, in) << "Unmatched input channel " << c_in << " and weight channel size "
                              << in << " with group size " << groups;

  tvm::tir::BijectiveLayout out_layout_converter(args->out_layout, "NCHW");
  tvm::Array<tvm::PrimExpr> oshape{tvm::Integer(n_in), tvm::Integer(out), tvm::Integer(h_out),
                                   tvm::Integer(w_out)};
  oshape = out_layout_converter.BackwardShape(oshape);

  call->out = TensorValue::Assemble(
      /*ctx=*/x->ctx,
      /*dtype=*/x->dtype,
      /*shape=*/
      {oshape[0].as<tvm::IntImmNode>()->value, oshape[1].as<tvm::IntImmNode>()->value,
       oshape[2].as<tvm::IntImmNode>()->value, oshape[3].as<tvm::IntImmNode>()->value});
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.conv2d", Conv2D).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

void Pool2D(const CallValues& call) {
  // NCHW
  const auto* args = call->args.as<PoolArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  CHECK_EQ(x->ndim, 4);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  std::vector<int64_t> stride = args->stride.empty() ? kernel : Pad<2>(args->stride);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  tvm::tir::BijectiveLayout layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> ishape{tvm::Integer(x->shape[0]), tvm::Integer(x->shape[1]),
                                   tvm::Integer(x->shape[2]), tvm::Integer(x->shape[3])};
  ishape = layout_converter.ForwardShape(ishape);
  int64_t n_in = ishape[0].as<tvm::IntImmNode>()->value;
  int64_t c_in = ishape[1].as<tvm::IntImmNode>()->value;
  int64_t h_in = ishape[2].as<tvm::IntImmNode>()->value;
  int64_t w_in = ishape[3].as<tvm::IntImmNode>()->value;
  int64_t kernel_h = kernel[0];
  int64_t kernel_w = kernel[1];
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h;
  int64_t pad_w;
  GetPadHW(args->padding, &pad_h, &pad_w);
  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out, w_out;
  CHECK(dilate_h == 1 && dilate_w == 1) << "Pooling does not support dilation!";
  if (!args->ceil_mode) {
    h_out = (h_in + pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  tvm::Array<tvm::PrimExpr> oshape{tvm::Integer(n_in), tvm::Integer(c_in), tvm::Integer(h_out),
                                   tvm::Integer(w_out)};
  oshape = layout_converter.BackwardShape(oshape);
  call->out = TensorValue::Assemble(
      /*ctx=*/x->ctx,
      /*dtype=*/x->dtype,
      /*shape=*/
      {oshape[0].as<tvm::IntImmNode>()->value, oshape[1].as<tvm::IntImmNode>()->value,
       oshape[2].as<tvm::IntImmNode>()->value, oshape[3].as<tvm::IntImmNode>()->value});
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.max_pool2d", Pool2D).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);
MNM_OP_DECLARE("mnm.op.avg_pool2d", Pool2D).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

void Softmax(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  NormalizeAxis(args->axis, x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.softmax", Softmax).set_attr<TOpPattern>("TOpPattern", kOpaque);
MNM_OP_DECLARE("mnm.op.log_softmax", Softmax).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.batch_norm_train", [](const CallValues& call) {
  const auto* args = call->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  TensorValue y = TensorValue::Assemble(/*ctx=*/x->ctx,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape);
  TensorValue running_mean = Downcast<TensorValue>(args->running_mean).CreateView();
  TensorValue running_var = Downcast<TensorValue>(args->running_var).CreateView();
  call->out = TupleValue::make(tvm::Array<Value>({y, running_mean, running_var}));
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.batch_norm_infer", [](const CallValues& call) {
  // FIXME(@were): please fix this: bn-infer should only output y
  const auto* args = call->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  TensorValue y = TensorValue::Assemble(/*ctx=*/x->ctx,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape);
  call->out = y;
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Conv2dDxw(const CallValues& call) {
  const auto* args = call->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  CHECK(args->shape.defined());
  const DLTensor* x_or_w = args->x_or_w;
  call->out = TensorValue::Assemble(/*ctx=*/x_or_w->ctx,
                                    /*dtype=*/x_or_w->dtype,
                                    /*shape=*/args->shape.value());
  call->ctx = x_or_w->ctx;
}

MNM_OP_DECLARE("mnm.op.conv2d_dx", Conv2dDxw).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);
MNM_OP_DECLARE("mnm.op.conv2d_dw", Conv2dDxw).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

void Pool2DDx(const CallValues& call) {
  const auto* args = call->args.as<PoolDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.max_pool2d_dx", Pool2DDx)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);
MNM_OP_DECLARE("mnm.op.avg_pool2d_dx", Pool2DDx)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

void SoftmaxDx(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.softmax_dx", SoftmaxDx).set_attr<TOpPattern>("TOpPattern", kOpaque);
MNM_OP_DECLARE("mnm.op.log_softmax_dx", SoftmaxDx).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.batch_norm_train_dxwb", [](const CallValues& call) {
  const auto* args = call->args.as<BatchNormTrainDxwbArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> xshape(x->shape, x->shape + x->ndim);
  TensorValue dx = TensorValue::Assemble(/*ctx=*/x->ctx,
                                         /*dtype=*/x->dtype,
                                         /*shape=*/xshape);
  const DLTensor* w = args->w;
  std::vector<int64_t> wshape(w->shape, w->shape + w->ndim);
  TensorValue dw = TensorValue::Assemble(/*ctx=*/w->ctx,
                                         /*dtype=*/w->dtype,
                                         /*shape=*/wshape);
  TensorValue db = TensorValue::Assemble(/*ctx=*/w->ctx,
                                         /*dtype=*/w->dtype,
                                         /*shape=*/wshape);
  call->out = TupleValue::make(tvm::Array<Value>({dx, dw, db}));
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

void BiasAdd(const CallValues& call) {
  const auto* args = call->args.as<BiasAddArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* bias = args->bias;
  CHECK(bias->ndim == 1) << "bias should only have 1 dim";
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}

MNM_OP_DECLARE("mnm.op.bias_add", BiasAdd).set_attr<TOpPattern>("TOpPattern", kBroadcast);
;

void LayerNorm(const CallValues& call) {
  const auto* args = call->args.as<LayerNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}
MNM_OP_DECLARE("mnm.op.layer_norm", LayerNorm).set_attr<TOpPattern>("TOpPattern", kOpaque);

void LayerNormDx(const CallValues& call) {
  const auto* args = call->args.as<LayerNormDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->ctx = x->ctx;
}
MNM_OP_DECLARE("mnm.op.layer_norm_dx", LayerNormDx).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Pad(const CallValues& call) {
  const auto* args = call->args.as<PadArgs>();
  CHECK(args != nullptr);
  const DLTensor* data = args->x;

  CHECK(args->pad_width.size() % 2 == 0);
  // check that pad widths match lengths
  CHECK(data->ndim == args->pad_width.size() / 2)
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->ndim << " dimensions "
      << "and there are " << args->pad_width.size() / 2 << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<int64_t> oshape;
  for (size_t i = 0; i < args->pad_width.size(); i += 2) {
    auto width1 = args->pad_width[i];
    auto width2 = args->pad_width[i + 1];
    CHECK(width1 >= 0) << "Param width elements should be positive but first pad width at "
                       << "index " << i << " is " << width1 << ".";
    CHECK(width2 >= 0) << "Param width elements should be positive but first pad width at "
                       << "index " << i << " is " << width2 << ".";

    auto padding = width1 + width2;
    oshape.push_back(data->shape[i / 2] + padding);
  }

  call->out = TensorValue::Assemble(/*ctx=*/data->ctx,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/oshape);
  call->ctx = data->ctx;
}
MNM_OP_DECLARE("mnm.op.pad", Pad).set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace declare
}  // namespace op
}  // namespace mnm
