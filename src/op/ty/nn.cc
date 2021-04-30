/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/data_layout.h>
#include <tvm/runtime/container.h>
#include <tvm/ir/env_func.h>
#include <vector>
#include "mnm/type.h"
#include "mnm/op_utils.h"
#include "../schema/nn.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using tvm::Array;
using tvm::Downcast;
using tvm::FloatImm;
using tvm::Integer;
using tvm::PrimExpr;
using tvm::relay::TensorType;
using tvm::relay::TupleType;
using tvm::relay::Type;
using namespace mnm::value;
using namespace schema;
using namespace mnm::type;

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

MNM_OP_TYPE("mnm.op.conv2d", "Conv2d", Conv2DInfer);

Type Conv2DDxwInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  Array<PrimExpr> res;
  if (args->shape.defined()) {
    for (auto value : args->shape.value()) {
      res.push_back(Integer(value->value));
    }
    return TensorType(res, dy->dtype);
  } else {
    return IncompleteType(tvm::kType);
  }
}

MNM_OP_TYPE("mnm.op.conv2d_dw", "Conv2dDxw", Conv2DDxwInfer);
MNM_OP_TYPE("mnm.op.conv2d_dx", "Conv2dDxw", Conv2DDxwInfer);

Type Pool2DInfer(const CallValues& value) {
  using namespace tvm::tir;
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
  CHECK(dilation[0] == 1 && dilation[1] == 1) << "Pooling does not support dilation!";
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

MNM_OP_TYPE("mnm.op.max_pool2d", "Pool2D", Pool2DInfer);
MNM_OP_TYPE("mnm.op.avg_pool2d", "Pool2D", Pool2DInfer);

Type AdaptivePool2DInfer(const CallValues& value) {
  const auto* args = value->args.as<AdaptivePoolArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  tvm::tir::BijectiveLayout data_layout_converter(args->layout, "NCHW");
  tvm::Array<tvm::PrimExpr> in_shape = data_layout_converter.ForwardShape(x->shape);
  std::vector<PrimExpr> oshape{in_shape[0], in_shape[1], Integer(args->shape[0]),
                               Integer(args->shape[1])};
  return TensorType(data_layout_converter.BackwardShape(oshape), x->dtype);
}

MNM_OP_TYPE("mnm.op.adaptive_max_pool2d", "AdaptivePool2D", AdaptivePool2DInfer);
MNM_OP_TYPE("mnm.op.adaptive_avg_pool2d", "AdaptivePool2D", AdaptivePool2DInfer);

MNM_OP_TYPE("mnm.op.max_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);
MNM_OP_TYPE("mnm.op.avg_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);
MNM_OP_TYPE("mnm.op.adaptive_max_pool2d_dx", "AdaptivePool2DDx",
            GeneralDxInfer<AdaptivePoolDxArgs>);
MNM_OP_TYPE("mnm.op.adaptive_avg_pool2d_dx", "AdaptivePool2DDx",
            GeneralDxInfer<AdaptivePoolDxArgs>);

Type BatchNormInferInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.batch_norm_infer", "BatchNormInfer", BatchNormInferInfer);

Type BatchNormTrainInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType running_mean = Downcast<TensorType>(GetType(args->running_mean));
  TensorType running_var = Downcast<TensorType>(GetType(args->running_var));
  return TupleType({x, running_mean, running_var});
}

MNM_OP_TYPE("mnm.op.batch_norm_train", "BatchNormTrain", BatchNormTrainInfer);

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

MNM_OP_TYPE("mnm.op.softmax", "Softmax", GeneralAxisInfer<SoftmaxArgs>);
MNM_OP_TYPE("mnm.op.log_softmax", "LogSoftmax", GeneralAxisInfer<SoftmaxArgs>);
MNM_OP_TYPE("mnm.op.softmax_dx", "SoftmaxDx", GeneralDxInfer<SoftmaxDxArgs>);
MNM_OP_TYPE("mnm.op.log_softmax_dx", "LogSoftmaxDx", GeneralDxInfer<SoftmaxDxArgs>);

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

MNM_OP_TYPE("mnm.op.batch_norm_train_dxwb", "BatchNormTrainDxwb", BatchNormTrainDxwbInfer);

Type BiasAddInfer(const CallValues& value) {
  const auto* args = value->args.as<BiasAddArgs>();
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.bias_add", "BiasAdd", BiasAddInfer);

Type ContribDropoutInfer(const CallValues& value) {
  const auto* args = value->args.as<DropoutArgs>();
  Array<Type> res;
  TensorType x_ty = Downcast<TensorType>(GetType(args->x));
  TensorType states_ty;
  if (args->in_states.defined()) {
    states_ty = Downcast<TensorType>(GetType(args->in_states.value()));
  } else {
    std::vector<PrimExpr> states_shape;
    states_ty = TensorType(states_shape, DataType::UInt(8));
  }
  res.push_back(x_ty);
  res.push_back(TensorType(x_ty->shape, DataType::Float(32)));
  res.push_back(states_ty);
  return TupleType(res);
}

MNM_OP_TYPE("mnm.op._contrib_dropout", "ContribDropout", ContribDropoutInfer);

MNM_OP_TYPE("mnm.op.layer_norm", "LayerNorm", GeneralAxisInfer<LayerNormArgs>);

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

MNM_OP_TYPE("mnm.op.layer_norm_dx", "LayerNormDx", LayerNormDxbInfer);

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

MNM_OP_TYPE("mnm.op.pad", "Pad", PadInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
