/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include <tvm/tir/data_layout.h>
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/tensor.h"
#include "raf/device.h"
#include "raf/device_api.h"
#include "raf/registry.h"
#include "../schema/nn.h"
#include "../ty/utils.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::ir;
using namespace raf::value;

void Conv2D(const CallValues& call) {
  // N.B.: NCHW + OIHW
  const auto* args = call->args.as<ConvArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* w = args->w;
  CHECK_EQ(x->ndim, 4);
  CHECK_EQ(w->ndim, 4);
  // TODO(@junrushao1994): deduce ctx here
  std::vector<int64_t> stride = raf::op::Pad<2>(args->stride);
  std::vector<int64_t> dilation = raf::op::Pad<2>(args->dilation);

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
      /*dev=*/x->device,
      /*dtype=*/x->dtype,
      /*shape=*/
      {oshape[0].as<tvm::IntImmNode>()->value, oshape[1].as<tvm::IntImmNode>()->value,
       oshape[2].as<tvm::IntImmNode>()->value, oshape[3].as<tvm::IntImmNode>()->value});
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.conv2d", Conv2D);

void Conv2dTrans(const CallValues& call) {
  // N.B.: NCHW + IOHW
  const auto* args = call->args.as<ConvTransArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* w = args->w;
  CHECK_EQ(x->ndim, 4);
  CHECK_EQ(w->ndim, 4);
  std::vector<int64_t> stride = raf::op::Pad<2>(args->stride);
  std::vector<int64_t> dilation = raf::op::Pad<2>(args->dilation);

  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape{
      tvm::Integer(x->shape[0]),
      tvm::Integer(x->shape[1]),
      tvm::Integer(x->shape[2]),
      tvm::Integer(x->shape[3]),
  };
  tvm::tir::BijectiveLayout w_layout_converter(args->kernel_layout, "IOHW");
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
  int64_t out = w_shape[1].as<tvm::IntImmNode>()->value;
  int64_t in = w_shape[0].as<tvm::IntImmNode>()->value;
  int64_t kernel_h = w_shape[2].as<tvm::IntImmNode>()->value;
  int64_t kernel_w = w_shape[3].as<tvm::IntImmNode>()->value;
  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h;
  int64_t pad_w;
  GetPadHW(args->padding, &pad_h, &pad_w);

  int64_t output_padding_h;
  int64_t output_padding_w;

  GetOutputPadHW(args->output_padding, &output_padding_h, &output_padding_w);

  int64_t dilate_h = dilation[0];
  int64_t dilate_w = dilation[1];
  CHECK(dilate_h == 1 && dilate_w == 1)
      << "Only supports dilation (1,1) but got  (" << dilate_h << "," << dilate_w << ")";
  CHECK((output_padding_h < stride_h && output_padding_w < stride_w) ||
        (output_padding_h < dilate_h && output_padding_w < dilate_w))
      << "Output padding must be smaller than either stride or dilation";

  int64_t h_out = (h_in - 1) * stride_h - pad_h + dilate_h * (kernel_h - 1) + output_padding_h + 1;
  int64_t w_out = (w_in - 1) * stride_w - pad_w + dilate_w * (kernel_w - 1) + output_padding_w + 1;

  int64_t groups = args->groups;
  CHECK_EQ(c_in / groups, in) << "Unmatched input channel " << c_in << " and weight channel size "
                              << in << " with group size " << groups;

  tvm::tir::BijectiveLayout out_layout_converter(args->out_layout, "NCHW");
  tvm::Array<tvm::PrimExpr> oshape{tvm::Integer(n_in), tvm::Integer(out), tvm::Integer(h_out),
                                   tvm::Integer(w_out)};
  oshape = out_layout_converter.BackwardShape(oshape);

  call->out = TensorValue::Assemble(
      /*dev=*/x->device,
      /*dtype=*/x->dtype,
      /*shape=*/
      {oshape[0].as<tvm::IntImmNode>()->value, oshape[1].as<tvm::IntImmNode>()->value,
       oshape[2].as<tvm::IntImmNode>()->value, oshape[3].as<tvm::IntImmNode>()->value});
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.conv2d_transpose", Conv2dTrans);

void Pool2D(const CallValues& call) {
  // NCHW
  const auto* args = call->args.as<PoolArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  CHECK_EQ(x->ndim, 4);
  std::vector<int64_t> kernel = raf::op::Pad<2>(args->kernel);
  std::vector<int64_t> stride = args->stride.empty() ? kernel : raf::op::Pad<2>(args->stride);
  std::vector<int64_t> dilation = raf::op::Pad<2>(args->dilation);
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
  if (!args->ceil_mode) {
    h_out = (h_in + pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = std::ceil((h_in + pad_h - dilate_h * (kernel_h - 1) - 1) /
                      static_cast<double_t>(stride_h)) +
            1;
    w_out = std::ceil((w_in + pad_w - dilate_w * (kernel_w - 1) - 1) /
                      static_cast<double_t>(stride_w)) +
            1;
  }
  tvm::Array<tvm::PrimExpr> oshape{tvm::Integer(n_in), tvm::Integer(c_in), tvm::Integer(h_out),
                                   tvm::Integer(w_out)};
  oshape = layout_converter.BackwardShape(oshape);
  call->out = TensorValue::Assemble(
      /*dev=*/x->device,
      /*dtype=*/x->dtype,
      /*shape=*/
      {oshape[0].as<tvm::IntImmNode>()->value, oshape[1].as<tvm::IntImmNode>()->value,
       oshape[2].as<tvm::IntImmNode>()->value, oshape[3].as<tvm::IntImmNode>()->value});
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.max_pool2d", Pool2D);
RAF_OP_DECLARE("raf.op.avg_pool2d", Pool2D);

void AdaptivePool2D(const CallValues& call) {
  const auto* args = call->args.as<AdaptivePoolArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  CHECK_EQ(x->ndim, 4);
  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape;
  for (int i = 0; i < x->ndim; ++i) {
    in_shape.push_back(tvm::Integer(x->shape[i]));
  }
  in_shape = data_layout_converter.ForwardShape(in_shape);
  tvm::Array<tvm::PrimExpr> out_shape{in_shape[0], in_shape[1], Integer(args->shape[0]),
                                      Integer(args->shape[1])};
  out_shape = data_layout_converter.BackwardShape(out_shape);
  std::vector<int64_t> out;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    const auto* s = out_shape[i].as<IntImmNode>();
    CHECK(s != nullptr);
    out.push_back(s->value);
  }
  call->out = TensorValue::Assemble(
      /*dev=*/x->device,
      /*dtype=*/x->dtype,
      /*shape=*/out);
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.adaptive_max_pool2d", AdaptivePool2D);
RAF_OP_DECLARE("raf.op.adaptive_avg_pool2d", AdaptivePool2D);

void Softmax(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  NormalizeAxis(args->axis, x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.softmax", Softmax);
RAF_OP_DECLARE("raf.op.log_softmax", Softmax);

RAF_OP_DECLARE("raf.op.batch_norm_train", [](const CallValues& call) {
  const auto* args = call->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  TensorValue y = TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape);
  TensorValue running_mean = Downcast<TensorValue>(args->running_mean);
  std::vector<int64_t> running_mean_shape(running_mean->tensor.Shape().begin(),
                                          running_mean->tensor.Shape().end());
  running_mean = running_mean.CreateView(running_mean_shape);
  TensorValue running_var = Downcast<TensorValue>(args->running_var);
  std::vector<int64_t> running_var_shape(running_var->tensor.Shape().begin(),
                                         running_var->tensor.Shape().end());
  running_var = running_var.CreateView(running_var_shape);
  call->out = TupleValue::make(tvm::Array<Value>({y, running_mean, running_var}));
  call->device = x->device;
}).set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{1, 1}, {2, 2}});

RAF_OP_DECLARE("raf.op.batch_norm_infer", [](const CallValues& call) {
  // FIXME(@were): please fix this: bn-infer should only output y
  const auto* args = call->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  TensorValue y = TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape);
  call->out = y;
  call->device = x->device;
});

void Conv2dDxw(const CallValues& call) {
  const auto* args = call->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  CHECK(args->shape.defined());
  const DLTensor* x_or_w = args->x_or_w;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  call->out = TensorValue::Assemble(/*dev=*/x_or_w->device,
                                    /*dtype=*/x_or_w->dtype,
                                    /*shape=*/shape);
  call->device = x_or_w->device;
}

RAF_OP_DECLARE("raf.op.conv2d_dx", Conv2dDxw);
RAF_OP_DECLARE("raf.op.conv2d_dw", Conv2dDxw);

void Conv2dTransposeDxw(const CallValues& call) {
  const auto* args = call->args.as<ConvTransposeDxwArgs>();
  CHECK(args != nullptr);
  CHECK(args->shape.defined());
  const DLTensor* x_or_w = args->x_or_w;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  call->out = TensorValue::Assemble(/*dev=*/x_or_w->device,
                                    /*dtype=*/x_or_w->dtype,
                                    /*shape=*/shape);
  call->device = x_or_w->device;
}

RAF_OP_DECLARE("raf.op.conv2d_transpose_dx", Conv2dTransposeDxw);
RAF_OP_DECLARE("raf.op.conv2d_transpose_dw", Conv2dTransposeDxw);

RAF_OP_DECLARE("raf.op.max_pool2d_dx", DeclareGeneralDx<PoolDxArgs>);
RAF_OP_DECLARE("raf.op.avg_pool2d_dx", DeclareGeneralDx<PoolDxArgs>);
RAF_OP_DECLARE("raf.op.adaptive_max_pool2d_dx", DeclareGeneralDx<AdaptivePoolDxArgs>);
RAF_OP_DECLARE("raf.op.adaptive_avg_pool2d_dx", DeclareGeneralDx<AdaptivePoolDxArgs>);
RAF_OP_DECLARE("raf.op.softmax_dx", DeclareGeneralDx<SoftmaxDxArgs>);
RAF_OP_DECLARE("raf.op.log_softmax_dx", DeclareGeneralDx<SoftmaxDxArgs>);

RAF_OP_DECLARE("raf.op.batch_norm_train_dxwb", [](const CallValues& call) {
  const auto* args = call->args.as<BatchNormTrainDxwbArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> xshape(x->shape, x->shape + x->ndim);
  TensorValue dx = TensorValue::Assemble(/*dev=*/x->device,
                                         /*dtype=*/x->dtype,
                                         /*shape=*/xshape);
  const DLTensor* w = args->w;
  std::vector<int64_t> wshape(w->shape, w->shape + w->ndim);
  TensorValue dw = TensorValue::Assemble(/*dev=*/w->device,
                                         /*dtype=*/w->dtype,
                                         /*shape=*/wshape);
  TensorValue db = TensorValue::Assemble(/*dev=*/w->device,
                                         /*dtype=*/w->dtype,
                                         /*shape=*/wshape);
  call->out = TupleValue::make(tvm::Array<Value>({dx, dw, db}));
  call->device = x->device;
});

void BiasAdd(const CallValues& call) {
  const auto* args = call->args.as<BiasAddArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const DLTensor* bias = args->bias;
  CHECK(bias->ndim == 1) << "bias should only have 1 dim";
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.bias_add", BiasAdd);

template <bool include_mask, bool include_reserve_space>
void ContribDropout(const CallValues& call) {
  const auto* args = call->args.as<DropoutArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  std::vector<int64_t> states_shape;
  std::vector<int64_t> reserve_space_shape;
  // The CUDNN compute generates reserve_space for backward usage.
#ifdef RAF_USE_CUDA
  const tvm::runtime::PackedFunc* pf =
      tvm::runtime::Registry::Get("raf.backend.cudnn.GetDropoutReserveSpaceSizeInBytes");
  if (include_reserve_space && pf) {
    Integer reserve_space_size_in_bytes = (*pf)(GetType(args->x));
    reserve_space_shape.push_back(reserve_space_size_in_bytes->value);
  }
#endif
  if (args->in_states.defined()) {
    const DLTensor* in_states = args->in_states.value();
    for (size_t i = 0; i < in_states->ndim; i++) {
      states_shape.push_back(tvm::Integer(in_states->shape[i]));
    }
  }
  TensorValue output = TensorValue::Assemble(/*dev=*/x->device,
                                             /*dtype=*/x->dtype,
                                             /*shape=*/shape);
  // valid for tvm only
  std::vector<int64_t> mask_shape;
  if (include_mask) {
    mask_shape = shape;
  }
  TensorValue mask = TensorValue::Assemble(/*dev=*/x->device,
                                           /*dtype=*/DType(DTypeCode::kFloat(), 32),
                                           /*shape=*/mask_shape);
  // valid for cudnn only
  TensorValue out_states = TensorValue::Assemble(/*dev=*/x->device,
                                                 /*dtype=*/DType(DTypeCode::kUInt(), 8),
                                                 /*shape=*/states_shape);
  // valid for cudnn only
  TensorValue reserve_space = TensorValue::Assemble(/*dev=*/x->device,
                                                    /*dtype=*/DType(DTypeCode::kUInt(), 8),
                                                    /*shape=*/reserve_space_shape);
  call->out = TupleValue::make(tvm::Array<Value>({output, mask, out_states, reserve_space}));
  call->device = x->device;
}

static const auto ContribDropoutBase = ContribDropout<true, true>;
static const auto ContribDropoutTVM = ContribDropout<true, false>;
static const auto ContribDropoutCudnn = ContribDropout<false, true>;
RAF_OP_DECLARE("raf.op._contrib_dropout", ContribDropoutBase);
RAF_OP_DECLARE("raf.op.tvm._contrib_dropout", ContribDropoutTVM);
RAF_OP_DECLARE("raf.op.cudnn._contrib_dropout", ContribDropoutCudnn);

void DropoutDx(const CallValues& call) {
  const auto* args = call->args.as<DropoutDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* dy = args->dy;
  std::vector<int64_t> shape(dy->shape, dy->shape + dy->ndim);
  call->out = TensorValue::Assemble(/*dev=*/dy->device,
                                    /*dtype=*/dy->dtype,
                                    /*shape=*/shape);
  call->device = dy->device;
}

RAF_OP_DECLARE("raf.op._contrib_dropout_dx", DropoutDx);

void LayerNorm(const CallValues& call) {
  const auto* args = call->args.as<LayerNormArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}
RAF_OP_DECLARE("raf.op.layer_norm", LayerNorm);

void LayerNormDx(const CallValues& call) {
  const auto* args = call->args.as<LayerNormDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> xshape(x->shape, x->shape + x->ndim);
  TensorValue dx = TensorValue::Assemble(/*dev=*/x->device,
                                         /*dtype=*/x->dtype,
                                         /*shape=*/xshape);
  if (args->scale.defined()) {
    const DLTensor* w = args->scale.value();
    std::vector<int64_t> wshape(w->shape, w->shape + w->ndim);

    TensorValue dw = TensorValue::Assemble(/*dev=*/w->device,
                                           /*dtype=*/w->dtype,
                                           /*shape=*/wshape);
    TensorValue db = TensorValue::Assemble(/*dev=*/w->device,
                                           /*dtype=*/w->dtype,
                                           /*shape=*/wshape);
    call->out = TupleValue::make(tvm::Array<Value>({dx, dw, db}));
  } else {
    call->out = dx;
  }
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.layer_norm_dx", LayerNormDx);

void Threshold(const CallValues& call) {
  const auto* args = call->args.as<ThresholdArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;

  // stop generating ir code when the size of tensor is zero
  for (auto len : shape) {
    if (len == 0) {
      call->callee = ir::NullValue<OpValue>();
      break;
    }
  }
}

RAF_OP_DECLARE("raf.op.threshold", Threshold);

void ThresholdDx(const CallValues& call) {
  const auto* args = call->args.as<ThresholdDxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op.threshold_dx", ThresholdDx);

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

  call->out = TensorValue::Assemble(/*dev=*/data->device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/oshape);
  call->device = data->device;
}
RAF_OP_DECLARE("raf.op.pad", Pad);

}  // namespace declare
}  // namespace op
}  // namespace raf
