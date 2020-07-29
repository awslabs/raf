/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/nn.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

template <int n>
static std::vector<int64_t> Pad(const std::vector<int64_t>& a) {
  int size = a.size();
  CHECK(size == 1 || size == n);
  return size == 1 ? std::vector<int64_t>(n, a[0]) : a;
}

MNM_OP_DECLARE("mnm.op.conv2d", [](const CallValues& call) {
  // N.B.: NCHW + OIHW
  const auto* args = call->args.as<ConvArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* w = args->w;
  CHECK_EQ(x->ndim, 4);
  CHECK_EQ(w->ndim, 4);
  // TODO(@junrushao1994): deduce ctx here
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  int64_t n_in = x->shape[0];
  int64_t c_in = x->shape[1];
  int64_t h_in = x->shape[2];
  int64_t w_in = x->shape[3];
  int64_t out = w->shape[0];
  int64_t in = w->shape[1];
  int64_t kernel_h = w->shape[2];
  int64_t kernel_w = w->shape[3];
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  int64_t w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  int64_t groups = args->groups;
  CHECK_EQ(c_in / groups, in) << "Unmatched input channel " << c_in
                              << " and weight channel size" << in
                              << " with group size " << groups;
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/{n_in, out, h_out, w_out});
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

void Pool2D(const CallValues& call) {
  // NCHW
  const auto* args = call->args.as<PoolArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  CHECK_EQ(x->ndim, 4);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  std::vector<int64_t> stride = args->stride.empty() ? kernel : Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  int64_t n_in = x->shape[0];
  int64_t c_in = x->shape[1];
  int64_t h_in = x->shape[2];
  int64_t w_in = x->shape[3];
  int64_t kernel_h = kernel[0];
  int64_t kernel_w = kernel[1];
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  int64_t h_out, w_out;
  CHECK(dilate_h == 1 && dilate_w == 1) << "Pooling does not support dilation!";
  if (!args->ceil_mode) {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/{n_in, c_in, h_out, w_out});
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
  TensorValue running_mean = TensorValue::make((args->running_mean).as<TensorValueObj>()->tensor
                                               .CreateView());
  TensorValue running_var = TensorValue::make((args->running_var).as<TensorValueObj>()->tensor
                                              .CreateView());
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
  const DLTensor* x_or_w = args->x_or_w;
  call->out = TensorValue::Assemble(/*ctx=*/x_or_w->ctx,
                                    /*dtype=*/x_or_w->dtype,
                                    /*shape=*/args->shape);
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

}  // namespace declare
}  // namespace op
}  // namespace mnm
