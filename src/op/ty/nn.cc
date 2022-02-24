/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/data_layout.h>
#include <tvm/ir/env_func.h>
#include <vector>
#include "raf/type.h"
#include "raf/op_utils.h"
#include "../schema/nn.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace schema;

Type Conv2DInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType w = Downcast<TensorType>(GetType(args->w));
  CHECK_EQ(x->shape.size(), 4) << x->shape;
  CHECK_EQ(w->shape.size(), 4) << w->shape;

  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);

  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape = x->shape;
  tvm::tir::BijectiveLayout w_layout_converter(args->kernel_layout, "OIHW");
  tvm::Array<tvm::PrimExpr> w_shape = w->shape;

  in_shape = data_layout_converter.ForwardShape(in_shape);
  w_shape = w_layout_converter.ForwardShape(w_shape);

  PrimExpr n_in = in_shape[0];
  PrimExpr c_in = in_shape[1];
  PrimExpr h_in = in_shape[2];
  PrimExpr w_in = in_shape[3];
  PrimExpr out = w_shape[0];
  PrimExpr in = w_shape[1];
  PrimExpr kernel_h = w_shape[2];
  PrimExpr kernel_w = w_shape[3];
  PrimExpr stride_h = Integer(stride[0]);
  PrimExpr stride_w = Integer(stride[1]);

  int64_t pad_h_int;
  int64_t pad_w_int;
  GetPadHW(args->padding, &pad_h_int, &pad_w_int);
  PrimExpr pad_h = Integer(pad_h_int);
  PrimExpr pad_w = Integer(pad_w_int);

  PrimExpr dilate_h = Integer(dilation[0]);
  PrimExpr dilate_w = Integer(dilation[1]);

  PrimExpr h_out = (h_in + pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  PrimExpr w_out = (w_in + pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;

  PrimExpr groups = Integer(args->groups);
  CHECK(TypeCheckCompare(c_in / groups, in, std::equal_to<int>()))
      << "Unmatched input channel " << c_in << " and weight channel size" << in
      << " with group size " << groups;

  tvm::tir::BijectiveLayout out_layout_converter(args->out_layout, "NCHW");
  tvm::Array<tvm::PrimExpr> oshape{n_in, out, h_out, w_out};
  oshape = out_layout_converter.BackwardShape(oshape);

  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.conv2d", "Conv2d", Conv2DInfer);

Type Conv2DTransInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvTransArgs>();
  CHECK(args != nullptr);

  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType w = Downcast<TensorType>(GetType(args->w));
  CHECK_EQ(x->shape.size(), 4) << x->shape;
  CHECK_EQ(w->shape.size(), 4) << w->shape;

  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);

  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape = x->shape;
  tvm::tir::BijectiveLayout w_layout_converter(args->kernel_layout, "IOHW");
  tvm::Array<tvm::PrimExpr> w_shape = w->shape;

  in_shape = data_layout_converter.ForwardShape(in_shape);
  w_shape = w_layout_converter.ForwardShape(w_shape);

  PrimExpr n_in = in_shape[0];
  PrimExpr c_in = in_shape[1];
  PrimExpr h_in = in_shape[2];
  PrimExpr w_in = in_shape[3];
  PrimExpr out = w_shape[1];
  PrimExpr in = w_shape[0];
  PrimExpr kernel_h = w_shape[2];
  PrimExpr kernel_w = w_shape[3];
  PrimExpr stride_h = Integer(stride[0]);
  PrimExpr stride_w = Integer(stride[1]);

  int64_t pad_h_int;
  int64_t pad_w_int;
  GetPadHW(args->padding, &pad_h_int, &pad_w_int);
  PrimExpr pad_h = Integer(pad_h_int);
  PrimExpr pad_w = Integer(pad_w_int);

  PrimExpr dilate_h = Integer(dilation[0]);
  PrimExpr dilate_w = Integer(dilation[1]);

  int64_t output_padding_h_int;
  int64_t output_padding_w_int;
  GetOutputPadHW(args->output_padding, &output_padding_h_int, &output_padding_w_int);

  PrimExpr output_padding_h = Integer(output_padding_h_int);
  PrimExpr output_padding_w = Integer(output_padding_w_int);

  PrimExpr h_out = (h_in - 1) * stride_h - pad_h + dilate_h * (kernel_h - 1) + output_padding_h + 1;
  PrimExpr w_out = (w_in - 1) * stride_w - pad_w + dilate_w * (kernel_w - 1) + output_padding_w + 1;

  PrimExpr groups = Integer(args->groups);
  CHECK(TypeCheckCompare(c_in / groups, in, std::equal_to<int>()))
      << "in nn.cc Unmatched input channel " << c_in << " and weight channel size " << in
      << " with group size " << groups;

  tvm::tir::BijectiveLayout out_layout_converter(args->out_layout, "NCHW");
  tvm::Array<tvm::PrimExpr> oshape{n_in, out, h_out, w_out};
  oshape = out_layout_converter.BackwardShape(oshape);
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.conv2d_transpose", "Conv2dTans", Conv2DTransInfer);

Type Conv2DDxwInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  if (args->shape.defined()) {
    Array<PrimExpr> shape = GetShapeExprFromValue(args->shape);
    return TensorType(shape, dy->dtype);
  } else {
    return IncompleteType(tvm::kType);
  }
}

RAF_OP_TYPE("raf.op.conv2d_dw", "Conv2dDxw", Conv2DDxwInfer);
RAF_OP_TYPE("raf.op.conv2d_dx", "Conv2dDxw", Conv2DDxwInfer);

Type Conv2DTransposeDxwInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvTransposeDxwArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  if (args->shape.defined()) {
    Array<PrimExpr> shape = GetShapeExprFromValue(args->shape);
    return TensorType(shape, dy->dtype);
  } else {
    return IncompleteType(tvm::kType);
  }
}

RAF_OP_TYPE("raf.op.conv2d_transpose_dw", "Conv2dTransposeDxw", Conv2DTransposeDxwInfer);
RAF_OP_TYPE("raf.op.conv2d_transpose_dx", "Conv2dTransposeDxw", Conv2DTransposeDxwInfer);

Type Pool2DInfer(const CallValues& value) {
  const auto* args = value->args.as<PoolArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  std::vector<int64_t> stride = args->stride.empty() ? kernel : Pad<2>(args->stride);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  tvm::tir::BijectiveLayout layout_converter(args->layout, "NCHW");
  Array<PrimExpr> shape = layout_converter.ForwardShape(x->shape);
  PrimExpr n_in = shape[0];
  PrimExpr c_in = shape[1];
  PrimExpr h_in = shape[2];
  PrimExpr w_in = shape[3];
  PrimExpr kernel_h = Integer(kernel[0]);
  PrimExpr kernel_w = Integer(kernel[1]);
  PrimExpr stride_h = Integer(stride[0]);
  PrimExpr stride_w = Integer(stride[1]);
  int64_t pad_h_int;
  int64_t pad_w_int;
  GetPadHW(args->padding, &pad_h_int, &pad_w_int);
  PrimExpr pad_h = Integer(pad_h_int);
  PrimExpr pad_w = Integer(pad_w_int);
  PrimExpr dilate_h = Integer(dilation[0]);
  PrimExpr dilate_w = Integer(dilation[1]);
  PrimExpr h_out, w_out;
  PrimExpr h_temp = (h_in + pad_h - dilate_h * (kernel_h - 1) - 1);
  PrimExpr w_temp = (w_in + pad_w - dilate_w * (kernel_w - 1) - 1);
  if (!args->ceil_mode) {
    h_out = h_temp / stride_h + 1;
    w_out = w_temp / stride_w + 1;
  } else {
    h_out = h_temp / stride_h + if_then_else(indexmod(h_temp, stride_h) == 0, 1, 2);
    w_out = w_temp / stride_w + if_then_else(indexmod(w_temp, stride_w) == 0, 1, 2);
  }
  Array<PrimExpr> oshape{n_in, c_in, h_out, w_out};
  return TensorType(layout_converter.BackwardShape(oshape), x->dtype);
}

RAF_OP_TYPE("raf.op.max_pool2d", "Pool2D", Pool2DInfer);
RAF_OP_TYPE("raf.op.avg_pool2d", "Pool2D", Pool2DInfer);

Type AdaptivePool2DInfer(const CallValues& value) {
  const auto* args = value->args.as<AdaptivePoolArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape = data_layout_converter.ForwardShape(x->shape);
  std::vector<PrimExpr> oshape{in_shape[0], in_shape[1], Integer(args->shape[0]),
                               Integer(args->shape[1])};
  return TensorType(data_layout_converter.BackwardShape(oshape), x->dtype);
}

RAF_OP_TYPE("raf.op.adaptive_max_pool2d", "AdaptivePool2D", AdaptivePool2DInfer);
RAF_OP_TYPE("raf.op.adaptive_avg_pool2d", "AdaptivePool2D", AdaptivePool2DInfer);

RAF_OP_TYPE("raf.op.max_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);
RAF_OP_TYPE("raf.op.avg_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);
RAF_OP_TYPE("raf.op.adaptive_max_pool2d_dx", "AdaptivePool2DDx",
            GeneralDxInfer<AdaptivePoolDxArgs>);
RAF_OP_TYPE("raf.op.adaptive_avg_pool2d_dx", "AdaptivePool2DDx",
            GeneralDxInfer<AdaptivePoolDxArgs>);

Type BatchNormInferInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

RAF_OP_TYPE("raf.op.batch_norm_infer", "BatchNormInfer", BatchNormInferInfer);

Type BatchNormTrainInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType running_mean = Downcast<TensorType>(GetType(args->running_mean));
  TensorType running_var = Downcast<TensorType>(GetType(args->running_var));
  return TupleType({x, running_mean, running_var});
}

RAF_OP_TYPE("raf.op.batch_norm_train", "BatchNormTrain", BatchNormTrainInfer);

template <typename T>
Type GeneralAxisInfer(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int axis = args->axis;
  int ndim = x->shape.size();
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return x;
}

RAF_OP_TYPE("raf.op.softmax", "Softmax", GeneralAxisInfer<SoftmaxArgs>);
RAF_OP_TYPE("raf.op.log_softmax", "LogSoftmax", GeneralAxisInfer<SoftmaxArgs>);
RAF_OP_TYPE("raf.op.softmax_dx", "SoftmaxDx", GeneralDxInfer<SoftmaxDxArgs>);
RAF_OP_TYPE("raf.op.log_softmax_dx", "LogSoftmaxDx", GeneralDxInfer<SoftmaxDxArgs>);

Type BatchNormTrainDxwbInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormTrainDxwbArgs>();
  CHECK(args != nullptr);
  TensorType dx = Downcast<TensorType>(GetType(args->x));
  TensorType dw = Downcast<TensorType>(GetType(args->w));
  TensorType db = Downcast<TensorType>(GetType(args->b));
  Array<Type> res;
  res.push_back(dx);
  res.push_back(dw);
  res.push_back(db);
  return TupleType(res);
}

RAF_OP_TYPE("raf.op.batch_norm_train_dxwb", "BatchNormTrainDxwb", BatchNormTrainDxwbInfer);

Type BiasAddInfer(const CallValues& value) {
  const auto* args = value->args.as<BiasAddArgs>();
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.bias_add", "BiasAdd", BiasAddInfer);

template <bool include_mask, bool include_reserve_space>
Type ContribDropoutInfer(const CallValues& value) {
  const auto* args = value->args.as<DropoutArgs>();
  TensorType x_ty = Downcast<TensorType>(GetType(args->x));
  TensorType reserve_space({}, DataType::UInt(8));
#ifdef RAF_USE_CUDA
  const tvm::runtime::PackedFunc* pf =
      tvm::runtime::Registry::Get("raf.backend.cudnn.GetDropoutReserveSpaceSizeInBytes");
  if (include_reserve_space && pf) {
    Integer reserve_space_size_in_bytes = (*pf)(GetType(args->x));
    Array<PrimExpr> reserve_space_shape = {reserve_space_size_in_bytes};
    reserve_space = TensorType(reserve_space_shape, DataType::UInt(8));
  }
#endif
  TensorType states_ty;
  if (args->in_states.defined()) {
    states_ty = Downcast<TensorType>(GetType(args->in_states.value()));
  } else {
    states_ty = TensorType({}, DataType::UInt(8));
  }
  Array<PrimExpr> mask_shape;
  if (include_mask) {
    mask_shape = x_ty->shape;
  }
  TensorType mask_ty(mask_shape, DataType::Float(32));
  return TupleType(Array<Type>{x_ty, mask_ty, states_ty, reserve_space});
}

static const auto ContribDropoutBase = ContribDropoutInfer<true, true>;
static const auto ContribDropoutTVM = ContribDropoutInfer<true, false>;
static const auto ContribDropoutCudnn = ContribDropoutInfer<false, true>;
RAF_OP_TYPE("raf.op._contrib_dropout", "ContribDropout", ContribDropoutBase);
RAF_OP_TYPE("raf.op.tvm._contrib_dropout", "ContribDropoutTVM", ContribDropoutTVM);
RAF_OP_TYPE("raf.op.cudnn._contrib_dropout", "ContribDropoutCudnn", ContribDropoutCudnn);

Type ContribDropoutDxInfer(const CallValues& value) {
  const auto* args = value->args.as<DropoutDxArgs>();
  return GetType(args->dy);
}

RAF_OP_TYPE("raf.op._contrib_dropout_dx", "ContribDropoutDx", ContribDropoutDxInfer);

RAF_OP_TYPE("raf.op.layer_norm", "LayerNorm", GeneralAxisInfer<LayerNormArgs>);

Type LayerNormDxbInfer(const CallValues& value) {
  const auto* args = value->args.as<LayerNormDxArgs>();
  CHECK(args != nullptr);
  TensorType dx = Downcast<TensorType>(GetType(args->x));
  if (args->scale.defined()) {
    TensorType dw = Downcast<TensorType>(GetType(args->scale.value()));
    Array<Type> res;
    res.push_back(dx);
    res.push_back(dw);
    res.push_back(dw);
    return TupleType(res);
  } else {
    return dx;
  }
}

RAF_OP_TYPE("raf.op.layer_norm_dx", "LayerNormDx", LayerNormDxbInfer);

Type ThresholdInfer(const CallValues& value) {
  const auto* args = value->args.as<ThresholdArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.threshold", "Threshold", ThresholdInfer);

Type ThresholdDxInfer(const CallValues& value) {
  const auto* args = value->args.as<ThresholdDxArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.threshold_dx", "ThresholdDx", ThresholdDxInfer);

Type PadInfer(const CallValues& value) {
  const auto* args = value->args.as<PadArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->x));

  CHECK(args->pad_width.size() % 2 == 0);
  // check that pad widths match lengths
  CHECK(data->shape.size() == args->pad_width.size() / 2)
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->shape.size() << " dimensions "
      << "and there are " << args->pad_width.size() / 2 << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<PrimExpr> oshape;
  for (size_t i = 0; i < args->pad_width.size(); i += 2) {
    auto width1 = args->pad_width[i];
    auto width2 = args->pad_width[i + 1];
    CHECK(width1 >= 0) << "Param width elements should be positive but first pad width at "
                       << "index " << i << " is " << width1 << ".";
    CHECK(width2 >= 0) << "Param width elements should be positive but first pad width at "
                       << "index " << i << " is " << width2 << ".";

    auto padding = Integer(width1 + width2);
    oshape.push_back(data->shape[i / 2] + padding);
  }

  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op.pad", "Pad", PadInfer);

}  // namespace op
}  // namespace raf
