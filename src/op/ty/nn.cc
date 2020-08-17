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

namespace mnm {
namespace op {
namespace type {

using tvm::Array;
using tvm::Downcast;
using tvm::Integer;
using tvm::PrimExpr;
using tvm::relay::TensorType;
using tvm::relay::Type;
using namespace mnm::value;
using schema::BatchNormArgs;
using schema::BiasAddArgs;
using schema::ConvArgs;
using schema::PoolArgs;
using schema::SoftmaxArgs;

template <int n>
static std::vector<int64_t> Pad(const std::vector<int64_t>& a) {
  int size = a.size();
  CHECK(size == 1 || size == n);
  return size == 1 ? std::vector<int64_t>(n, a[0]) : a;
}

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

Type BatchNormInferInfer(const CallValues& value) {
  const auto* args = value->args.as<BatchNormArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.batch_norm_infer", "BatchNormInfer", BatchNormInferInfer);

Type SoftmaxInfer(const CallValues& value) {
  const auto* args = value->args.as<SoftmaxArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

MNM_OP_TYPE("mnm.op.softmax", "Softmax", SoftmaxInfer);

using namespace mnm::value;
using schema::BiasAddArgs;
using tvm::relay::Type;

Type BiasAddInfer(const CallValues& value) {
  const auto* args = value->args.as<BiasAddArgs>();
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.bias_add", "BiasAdd", BiasAddInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
