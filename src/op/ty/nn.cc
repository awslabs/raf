/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/container.h>
#include <tvm/ir/env_func.h>
#include <vector>
#include "mnm/type.h"
#include "../schema/nn.h"
#include "./utils.h"
#include "../op_utils.h"

namespace mnm {
namespace op {
namespace type {

using schema::BatchNormArgs;
using schema::BiasAddArgs;
using schema::ConvArgs;
using schema::ConvDxwArgs;
using schema::PoolArgs;
using schema::SoftmaxArgs;
using tvm::Array;
using tvm::Downcast;
using tvm::Integer;
using tvm::PrimExpr;
using tvm::relay::TensorType;
using tvm::relay::Type;
using namespace mnm::value;
using namespace schema;
using namespace mnm::type;

Type Conv2DInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  TensorType w = Downcast<TensorType>(GetType(args->w));
  CHECK_EQ(x->shape.size(), 4);
  CHECK_EQ(w->shape.size(), 4);
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  PrimExpr n_in = x->shape[0];
  PrimExpr c_in = x->shape[1];
  PrimExpr h_in = x->shape[2];
  PrimExpr w_in = x->shape[3];
  PrimExpr out = w->shape[0];
  PrimExpr in = w->shape[1];
  PrimExpr kernel_h = w->shape[2];
  PrimExpr kernel_w = w->shape[3];
  PrimExpr stride_h = Integer(stride[0]);
  PrimExpr stride_w = Integer(stride[1]);
  PrimExpr pad_h = Integer(padding[0]);
  PrimExpr pad_w = Integer(padding[1]);
  PrimExpr dilate_h = Integer(dilation[0]);
  PrimExpr dilate_w = Integer(dilation[1]);
  PrimExpr h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
  PrimExpr w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  PrimExpr groups = Integer(args->groups);
  CHECK(TypeCheckEqual(c_in / groups, in))
      << "Unmatched input channel " << c_in << " and weight channel size" << in
      << " with group size " << groups;
  return TensorType(Array<PrimExpr>{n_in, out, h_out, w_out}, x->dtype);
}

MNM_OP_TYPE("mnm.op.conv2d", "Conv2d", Conv2DInfer);

Type Conv2DDxwInfer(const CallValues& value) {
  const auto* args = value->args.as<ConvDxwArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  std::vector<int64_t> shape = Pad<4>(args->shape);
  Array<PrimExpr> res;
  for (int i = 0; i < shape.size(); ++i) res.push_back(Integer(shape[i]));
  return TensorType(res, dy->dtype);
}

MNM_OP_TYPE("mnm.op.conv2d_dw", "Conv2dDxw", Conv2DDxwInfer);
MNM_OP_TYPE("mnm.op.conv2d_dx", "Conv2dDxw", Conv2DDxwInfer);

Type Pool2DInfer(const CallValues& value) {
  const auto* args = value->args.as<PoolArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  std::vector<int64_t> stride = args->stride.empty() ? kernel : Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  PrimExpr n_in = x->shape[0];
  PrimExpr c_in = x->shape[1];
  PrimExpr h_in = x->shape[2];
  PrimExpr w_in = x->shape[3];
  PrimExpr kernel_h = Integer(kernel[0]);
  PrimExpr kernel_w = Integer(kernel[1]);
  PrimExpr stride_h = Integer(stride[0]);
  PrimExpr stride_w = Integer(stride[1]);
  PrimExpr pad_h = Integer(padding[0]);
  PrimExpr pad_w = Integer(padding[1]);
  PrimExpr dilate_h = Integer(dilation[0]);
  PrimExpr dilate_w = Integer(dilation[1]);
  PrimExpr h_out, w_out;
  CHECK(dilation[0] == 1 && dilation[1] == 1) << "Pooling does not support dilation!";
  if (!args->ceil_mode) {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) - 1) / stride_w + 1;
  } else {
    h_out = (h_in + 2 * pad_h - dilate_h * (kernel_h - 1) + stride_h - 1) / stride_h + 1;
    w_out = (w_in + 2 * pad_w - dilate_w * (kernel_w - 1) + stride_w - 1) / stride_w + 1;
  }
  return TensorType(Array<PrimExpr>{n_in, c_in, h_out, w_out}, x->dtype);
}

MNM_OP_TYPE("mnm.op.max_pool2d", "Pool2D", Pool2DInfer);
MNM_OP_TYPE("mnm.op.avg_pool2d", "Pool2D", Pool2DInfer);

template <typename T>
Type GeneralDxInfer(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.max_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);
MNM_OP_TYPE("mnm.op.avg_pool2d_dx", "Pool2DDx", GeneralDxInfer<PoolDxArgs>);

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

Type BiasAddInfer(const CallValues& value) {
  const auto* args = value->args.as<BiasAddArgs>();
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.bias_add", "BiasAdd", BiasAddInfer);

MNM_OP_TYPE("mnm.op.layer_norm", "LayerNorm", GeneralAxisInfer<LayerNormArgs>);
MNM_OP_TYPE("mnm.op.layer_norm_dx", "LayerNormDx", GeneralDxInfer<LayerNormDxArgs>);

}  // namespace type
}  // namespace op
}  // namespace mnm
