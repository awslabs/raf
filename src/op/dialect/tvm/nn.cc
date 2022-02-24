/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/topi/nn.h>
#include <vector>
#include "raf/op_utils.h"
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace schema;

Attrs BinarySchema2DenseAttrs(const BinaryArgs* args) {
  auto attrs = make_object<tvm::relay::DenseAttrs>();
  return Attrs(attrs);
}

template <bool transpose_a, bool transpose_b>
Attrs BinarySchema2BatchMatmulAttrs(const BinaryArgs* args) {
  auto attrs = make_object<tvm::relay::BatchMatmulAttrs>();
  attrs->out_dtype = NullValue<DataType>();
  attrs->transpose_a = transpose_a;
  attrs->transpose_b = transpose_b;
  return Attrs(attrs);
}

RAF_TVM(matmul, Matmul, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        BinarySchema2DenseAttrs, GenericHasher, kOutEWiseFusable);
RAF_TVM(matmul_tn, MatmulTN, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        BinarySchema2DenseAttrs, GenericHasher, kOutEWiseFusable);
RAF_TVM(matmul_nt, MatmulNT, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        BinarySchema2DenseAttrs, GenericHasher, kOutEWiseFusable);
RAF_TVM(matmul_tt, MatmulTT, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        BinarySchema2DenseAttrs, GenericHasher, kOutEWiseFusable);
RAF_TVM(dense, Dense, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames, BinarySchema2DenseAttrs,
        GenericHasher, kOutEWiseFusable);
RAF_TVM(batch_matmul, BatchMatmul, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        (BinarySchema2BatchMatmulAttrs<false, false>), GenericHasher, kOutEWiseFusable);
RAF_TVM(batch_matmul_nt, BatchMatmulNT, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        (BinarySchema2BatchMatmulAttrs<false, true>), GenericHasher, kOutEWiseFusable);
RAF_TVM(batch_matmul_tn, BatchMatmulTN, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        (BinarySchema2BatchMatmulAttrs<true, false>), GenericHasher, kOutEWiseFusable);
RAF_TVM(batch_matmul_tt, BatchMatmulTT, BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
        (BinarySchema2BatchMatmulAttrs<true, true>), GenericHasher, kOutEWiseFusable);

std::vector<Value> ConvSchema2Args(const ConvArgs* args) {
  return {args->x, args->w};
}

std::vector<std::string> ConvSchemaArgNames(const op::CallValues& call) {
  return {"x", "w"};
}

Attrs ConvSchema2Attrs(const ConvArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<Conv2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->kernel_size = NullValue<Array<tvm::relay::IndexExpr>>();
  attrs->data_layout = args->layout;
  attrs->kernel_layout = args->kernel_layout;
  attrs->out_layout = args->out_layout;

  return Attrs(attrs);
}

HashKey Conv2dHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const ConvArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->dilation;
  key << args->groups;
  return key;
}

RAF_TVM(conv2d, Conv2d, ConvArgs, ConvSchema2Args, ConvSchemaArgNames, ConvSchema2Attrs,
        Conv2dHasher, kOutEWiseFusable);

std::vector<Value> ConvTransSchema2Args(const ConvTransArgs* args) {
  return {args->x, args->w};
}

std::vector<std::string> ConvTransSchemaArgNames(const op::CallValues& call) {
  return {"x", "w"};
}

Attrs ConvTransSchema2Attrs(const ConvTransArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
  std::vector<int64_t> output_padding =
      args->output_padding.size() > 1 ? args->output_padding : Pad<2>(args->output_padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<tvm::relay::Conv2DTransposeAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < output_padding.size(); ++i) {
    attrs->output_padding.push_back(IntImm(tvm::runtime::DataType::Int(64), output_padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->kernel_size = NullValue<Array<tvm::relay::IndexExpr>>();
  attrs->data_layout = args->layout;
  attrs->kernel_layout = args->kernel_layout;
  attrs->out_layout = args->out_layout;
  return Attrs(attrs);
}

HashKey Conv2dTransHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const ConvTransArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->output_padding;

  key << args->dilation;
  key << args->groups;
  return key;
}

RAF_TVM(conv2d_transpose, Conv2dTrans, ConvTransArgs, ConvTransSchema2Args, ConvTransSchemaArgNames,
        ConvTransSchema2Attrs, Conv2dTransHasher, kOutEWiseFusable);

std::vector<Value> ConvDxwSchema2Args(const ConvDxwArgs* args) {
  std::vector<Value> re;
  re.push_back(args->x_or_w);
  if (args->y.defined()) {
    re.push_back(args->y.value());
  }
  re.push_back(args->dy);
  return re;
}

std::vector<std::string> ConvDxwSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<ConvDxwArgs>();
  std::vector<std::string> ret;
  ret.push_back("x_or_w");
  if (args->y.defined()) {
    ret.push_back("y");
  }
  ret.push_back("dy");
  return ret;
}

Attrs ConvDxwSchema2Attrs(const ConvDxwArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<Conv2dDxwAttrs>();
  if (args->y.defined()) {
    attrs->use_output = true;
  } else {
    attrs->use_output = false;
  }
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  // FIXME: (workaround) we use kernel size to store the shape of X (for dx) or W (for dw)
  auto shape = GetShapeExprFromValue(args->shape);
  for (int i = 0; i < shape.size(); ++i) {
    attrs->kernel_size.push_back(shape[i]);
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->data_layout = "NCHW";
  attrs->kernel_layout = "OIHW";
  attrs->out_layout = "NCHW";

  return Attrs(attrs);
}

HashKey Conv2dDxwHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const ConvDxwArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->dilation;
  key << args->groups;
  return key;
}

RAF_TVM_PLEVEL(conv2d_dx, Conv2dDx, ConvDxwArgs, ConvDxwSchema2Args, ConvDxwSchemaArgNames,
               ConvDxwSchema2Attrs, Conv2dDxwHasher, kOutEWiseFusable, 9);
RAF_TVM_PLEVEL(conv2d_dw, Conv2dDw, ConvDxwArgs, ConvDxwSchema2Args, ConvDxwSchemaArgNames,
               ConvDxwSchema2Attrs, Conv2dDxwHasher, kOutEWiseFusable, 9);

std::vector<Value> ConvTransposeDxwSchema2Args(const ConvTransposeDxwArgs* args) {
  std::vector<Value> re;
  re.push_back(args->x_or_w);
  if (args->y.defined()) {
    re.push_back(args->y.value());
  }
  re.push_back(args->dy);
  return re;
}

std::vector<std::string> ConvTransposeDxwSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<ConvTransposeDxwArgs>();
  std::vector<std::string> ret;
  ret.push_back("x_or_w");
  if (args->y.defined()) {
    ret.push_back("y");
  }
  ret.push_back("dy");
  return ret;
}

Attrs ConvTransposeDxwSchema2Attrs(const ConvTransposeDxwArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> output_padding = Pad<2>(args->output_padding);
  std::vector<int64_t> dilation = Pad<2>(args->dilation);
  auto attrs = make_object<Conv2dTransposeDxwAttrs>();
  if (args->y.defined()) {
    attrs->use_output = true;
  } else {
    attrs->use_output = false;
  }
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < output_padding.size(); ++i) {
    attrs->output_padding.push_back(IntImm(tvm::runtime::DataType::Int(64), output_padding[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  // FIXME: (workaround) we use kernel size to store the shape of X (for dx) or W (for dw)
  auto shape = GetShapeExprFromValue(args->shape);
  for (int i = 0; i < shape.size(); ++i) {
    attrs->kernel_size.push_back(shape[i]);
  }
  attrs->groups = args->groups;
  attrs->channels = NullValue<tvm::relay::IndexExpr>();
  attrs->data_layout = "NCHW";
  attrs->kernel_layout = "IOHW";
  attrs->out_layout = "NCHW";

  return Attrs(attrs);
}

HashKey Conv2dTransposeDxwHasher(const std::vector<Type>& param_types, const Type& y_type,
                                 const ConvTransposeDxwArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->output_padding;
  key << args->dilation;
  key << args->groups;
  return key;
}

RAF_TVM_PLEVEL(conv2d_transpose_dx, Conv2dTransposeDx, ConvTransposeDxwArgs,
               ConvTransposeDxwSchema2Args, ConvTransposeDxwSchemaArgNames,
               ConvTransposeDxwSchema2Attrs, Conv2dTransposeDxwHasher, kOutEWiseFusable, 9);
RAF_TVM_PLEVEL(conv2d_transpose_dw, Conv2dTransposeDw, ConvTransposeDxwArgs,
               ConvTransposeDxwSchema2Args, ConvTransposeDxwSchemaArgNames,
               ConvTransposeDxwSchema2Attrs, Conv2dTransposeDxwHasher, kOutEWiseFusable, 9);

std::vector<Value> SoftmaxSchema2Args(const SoftmaxArgs* args) {
  return {args->x};
}

std::vector<std::string> SoftmaxSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SoftmaxSchema2Attrs(const SoftmaxArgs* args) {
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SoftmaxHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const SoftmaxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

std::vector<Value> SoftmaxDxSchema2Args(const SoftmaxDxArgs* args) {
  return {args->x, args->y, args->dy};
}

std::vector<std::string> SoftmaxDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "y", "dy"};
}

Attrs SoftmaxDxSchema2Attrs(const SoftmaxDxArgs* args) {
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SoftmaxDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const SoftmaxDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

RAF_TVM(softmax, Softmax, SoftmaxArgs, SoftmaxSchema2Args, SoftmaxSchemaArgNames,
        SoftmaxSchema2Attrs, SoftmaxHasher, kOpaque);
RAF_TVM(softmax_dx, SoftmaxDx, SoftmaxDxArgs, SoftmaxDxSchema2Args, SoftmaxDxSchemaArgNames,
        SoftmaxDxSchema2Attrs, SoftmaxDxHasher, kOpaque);
RAF_TVM(log_softmax, LogSoftmax, SoftmaxArgs, SoftmaxSchema2Args, SoftmaxSchemaArgNames,
        SoftmaxSchema2Attrs, SoftmaxHasher, kOpaque);
RAF_TVM(log_softmax_dx, LogSoftmaxDx, SoftmaxDxArgs, SoftmaxDxSchema2Args, SoftmaxDxSchemaArgNames,
        SoftmaxDxSchema2Attrs, SoftmaxDxHasher, kOpaque);

std::vector<Value> BiasAddSchema2Args(const BiasAddArgs* args) {
  return {args->x, args->bias};
}

std::vector<std::string> BiasAddSchemaArgNames(const op::CallValues& call) {
  return {"x", "bias"};
}

Attrs BiasAddSchema2Attrs(const BiasAddArgs* args) {
  auto attrs = make_object<tvm::relay::BiasAddAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey BiasAddHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const BiasAddArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

RAF_TVM(bias_add, BiasAdd, BiasAddArgs, BiasAddSchema2Args, BiasAddSchemaArgNames,
        BiasAddSchema2Attrs, BiasAddHasher, kBroadcast);

std::vector<Value> ContribDropoutSchema2Args(const DropoutArgs* args) {
  std::vector<Value> re;
  re.push_back(args->x);
  if (args->in_states.defined()) {
    re.push_back(args->in_states.value());
  }
  return re;
}

std::vector<std::string> ContribDropoutSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<DropoutArgs>();
  std::vector<std::string> ret;
  ret.push_back("x");
  if (args->in_states.defined()) {
    ret.push_back("in_states");
  }
  return ret;
}

Attrs ContribDropoutSchema2Attrs(const DropoutArgs* args) {
  LOG(INFO) << "TVM implementation of _contrib_dropout not support states return";
  auto attrs = make_object<DropoutAttrs>();
  attrs->rate = args->p;
  return Attrs(attrs);
}

HashKey ContribDropoutHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const DropoutArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->p;
  return key;
}

RAF_TVM(_contrib_dropout, Dropout, DropoutArgs, ContribDropoutSchema2Args,
        ContribDropoutSchemaArgNames, ContribDropoutSchema2Attrs, ContribDropoutHasher, kOpaque);

std::vector<Value> ContribDropoutDxSchema2Args(const DropoutDxArgs* args) {
  return {args->dy, args->mask};
}

std::vector<std::string> ContribDropoutDxSchemaArgNames(const op::CallValues& call) {
  return {"dy", "mask"};
}

Attrs ContribDropoutDxSchema2Attrs(const DropoutDxArgs* args) {
  auto attrs = make_object<DropoutAttrs>();
  attrs->rate = args->p;
  return Attrs(attrs);
}

HashKey ContribDropoutDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                               const DropoutDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->p;
  return key;
}

RAF_TVM(_contrib_dropout_dx, DropoutDx, DropoutDxArgs, ContribDropoutDxSchema2Args,
        ContribDropoutDxSchemaArgNames, ContribDropoutDxSchema2Attrs, ContribDropoutDxHasher,
        kOpaque);

template <typename T>
std::vector<Value> PoolSchema2Args(const T* args) {
  return {args->x};
}

std::vector<std::string> PoolSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs MaxPoolSchema2Attrs(const PoolArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> dilation =
      args->dilation.size() > 1 ? args->dilation : Pad<2>(args->dilation);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  auto attrs = make_object<tvm::relay::MaxPool2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < kernel.size(); ++i) {
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel[i]));
  }
  attrs->ceil_mode = args->ceil_mode;
  attrs->layout = args->layout;
  CHECK_EQ(args->include_pad, true);
  return Attrs(attrs);
}

Attrs AvgPoolSchema2Attrs(const PoolArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> dilation =
      args->dilation.size() > 1 ? args->dilation : Pad<2>(args->dilation);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  auto attrs = make_object<tvm::relay::AvgPool2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < dilation.size(); ++i) {
    attrs->dilation.push_back(IntImm(tvm::runtime::DataType::Int(64), dilation[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < kernel.size(); ++i) {
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel[i]));
  }
  attrs->ceil_mode = args->ceil_mode;
  attrs->count_include_pad = args->include_pad;
  attrs->layout = args->layout;
  return Attrs(attrs);
}

HashKey PoolHasher(const std::vector<Type>& param_types, const Type& y_type, const PoolArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->dilation;
  key << args->padding;
  key << args->kernel;
  key << args->ceil_mode;
  key << args->include_pad;
  return key;
}

RAF_TVM(max_pool2d, MaxPool2D, PoolArgs, PoolSchema2Args<PoolArgs>, PoolSchemaArgNames,
        MaxPoolSchema2Attrs, PoolHasher, kOutEWiseFusable);
RAF_TVM(avg_pool2d, AvgPool2D, PoolArgs, PoolSchema2Args<PoolArgs>, PoolSchemaArgNames,
        AvgPoolSchema2Attrs, PoolHasher, kOutEWiseFusable);

Attrs AdaptivePoolSchema2Attrs(const AdaptivePoolArgs* args) {
  const DLTensor* x = args->x;
  auto attrs = make_object<tvm::relay::AdaptivePool2DAttrs>();
  for (size_t i = 0; i < args->shape.size(); ++i) {
    attrs->output_size.push_back(Integer(args->shape[i]));
  }
  attrs->layout = args->layout;
  return Attrs(attrs);
}

HashKey AdaptivePoolHasher(const std::vector<Type>& param_types, const Type& y_type,
                           const AdaptivePoolArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  return key;
}

RAF_TVM(adaptive_max_pool2d, AdaptiveMaxPool2D, AdaptivePoolArgs, PoolSchema2Args<AdaptivePoolArgs>,
        PoolSchemaArgNames, AdaptivePoolSchema2Attrs, AdaptivePoolHasher, kOutEWiseFusable);
RAF_TVM(adaptive_avg_pool2d, AdaptiveAvgPool2D, AdaptivePoolArgs, PoolSchema2Args<AdaptivePoolArgs>,
        PoolSchemaArgNames, AdaptivePoolSchema2Attrs, AdaptivePoolHasher, kOutEWiseFusable);

template <typename T>
std::vector<Value> PoolDxSchema2Args(const T* args) {
  return {args->dy, args->x};
}

std::vector<std::string> PoolDxSchemaArgNames(const op::CallValues& call) {
  return {"dy", "x"};
}

Attrs AvgPoolDxSchema2Attrs(const PoolDxArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  auto attrs = make_object<tvm::relay::AvgPool2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < kernel.size(); ++i) {
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel[i]));
  }
  attrs->ceil_mode = args->ceil_mode;
  attrs->count_include_pad = args->include_pad;
  attrs->layout = "NCHW";
  return Attrs(attrs);
}

Attrs MaxPoolDxSchema2Attrs(const PoolDxArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = Pad<2>(args->padding);
  std::vector<int64_t> kernel = Pad<2>(args->kernel);
  auto attrs = make_object<tvm::relay::MaxPool2DAttrs>();
  for (int i = 0; i < stride.size(); ++i) {
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride[i]));
  }
  for (int i = 0; i < padding.size(); ++i) {
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding[i]));
  }
  for (int i = 0; i < kernel.size(); ++i) {
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel[i]));
  }
  attrs->ceil_mode = args->ceil_mode;
  attrs->layout = "NCHW";
  CHECK_EQ(args->include_pad, true);
  return Attrs(attrs);
}

HashKey PoolDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const PoolDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->kernel;
  key << args->ceil_mode;
  key << args->include_pad;
  return key;
}

RAF_TVM(avg_pool2d_dx, AvgPool2DDx, PoolDxArgs, PoolDxSchema2Args<PoolDxArgs>, PoolDxSchemaArgNames,
        AvgPoolDxSchema2Attrs, PoolDxHasher, kOutEWiseFusable);
RAF_TVM(max_pool2d_dx, MaxPool2DDx, PoolDxArgs, PoolDxSchema2Args<PoolDxArgs>, PoolDxSchemaArgNames,
        MaxPoolDxSchema2Attrs, PoolDxHasher, kOutEWiseFusable);

Attrs AdaptiveAvgPoolDxSchema2Attrs(const AdaptivePoolDxArgs* args) {
  const DLTensor* x = args->x;
  auto attrs = make_object<tvm::relay::AvgPool2DAttrs>();
  std::vector<int64_t> out_hw = args->shape;
  std::vector<int64_t> in_hw = {/*h=*/x->shape[2], /*w=*/x->shape[3]};
  CHECK_EQ(out_hw.size(), in_hw.size());
  for (size_t i = 0; i < out_hw.size(); ++i) {
    int64_t stride, kernel_size, padding;
    GetAdaptivePoolKernel(in_hw[i], out_hw[i], &kernel_size, &stride, &padding);
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride));
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding));
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel_size));
  }
  attrs->ceil_mode = false;
  attrs->count_include_pad = true;
  attrs->layout = "NCHW";
  return Attrs(attrs);
}

Attrs AdaptiveMaxPoolDxSchema2Attrs(const AdaptivePoolDxArgs* args) {
  const DLTensor* x = args->x;
  auto attrs = make_object<tvm::relay::MaxPool2DAttrs>();
  std::vector<int64_t> out_hw = args->shape;
  std::vector<int64_t> in_hw = {/*h=*/x->shape[2], /*w=*/x->shape[3]};
  CHECK_EQ(out_hw.size(), in_hw.size());
  for (size_t i = 0; i < out_hw.size(); ++i) {
    int64_t stride, kernel_size, padding;
    GetAdaptivePoolKernel(in_hw[i], out_hw[i], &kernel_size, &stride, &padding);
    attrs->strides.push_back(IntImm(tvm::runtime::DataType::Int(64), stride));
    attrs->padding.push_back(IntImm(tvm::runtime::DataType::Int(64), padding));
    attrs->pool_size.push_back(IntImm(tvm::runtime::DataType::Int(64), kernel_size));
  }
  attrs->ceil_mode = false;
  attrs->layout = "NCHW";
  return Attrs(attrs);
}

HashKey AdaptivePoolDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const AdaptivePoolDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  return key;
}

// FIXME: We currently use injective schedule for these 2 ops so they cannot be fused
// with others without tuning. To avoid the tuning overhead we now use kOpaque to avoid
// fusion. After fixing the schedule to really support fusing output elementwise ops, we should
// change their fusion pattern back to kOutEWiseFusable.
RAF_TVM(adaptive_avg_pool2d_dx, AdaptiveAvgPool2DDx, AdaptivePoolDxArgs,
        PoolDxSchema2Args<AdaptivePoolDxArgs>, PoolDxSchemaArgNames, AdaptiveAvgPoolDxSchema2Attrs,
        AdaptivePoolDxHasher, kOpaque);
RAF_TVM(adaptive_max_pool2d_dx, AdaptiveMaxPool2DDx, AdaptivePoolDxArgs,
        PoolDxSchema2Args<AdaptivePoolDxArgs>, PoolDxSchemaArgNames, AdaptiveMaxPoolDxSchema2Attrs,
        AdaptivePoolDxHasher, kOpaque);

std::vector<Value> LayerNormSchema2Args(const LayerNormArgs* args) {
  std::vector<Value> re;
  re.push_back(args->x);
  if (args->scale.defined()) {
    re.push_back(args->scale.value());
  }
  if (args->bias.defined()) {
    re.push_back(args->bias.value());
  }
  return re;
}

std::vector<std::string> LayerNormSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<LayerNormArgs>();
  std::vector<std::string> ret;
  ret.push_back("x");
  if (args->scale.defined()) {
    ret.push_back("scale");
  }
  if (args->scale.defined()) {
    ret.push_back("bias");
  }
  return ret;
}

Attrs LayerNormSchema2Attrs(const LayerNormArgs* args) {
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = args->axis;
  attrs->epsilon = args->eps;
  if (args->scale.defined()) {
    attrs->set_scale_bias = true;
  } else {
    attrs->set_scale_bias = false;
  }
  return Attrs(attrs);
}

HashKey LayerNormHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const LayerNormArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->eps;
  return key;
}

RAF_TVM(layer_norm, LayerNorm, LayerNormArgs, LayerNormSchema2Args, LayerNormSchemaArgNames,
        LayerNormSchema2Attrs, LayerNormHasher, kOpaque);

std::vector<Value> LayerNormDxSchema2Args(const LayerNormDxArgs* args) {
  std::vector<Value> re;
  re.push_back(args->x);
  if (args->scale.defined()) {
    re.push_back(args->scale.value());
  }
  re.push_back(args->dy);
  return re;
}

std::vector<std::string> LayerNormDxSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<LayerNormDxArgs>();
  std::vector<std::string> ret;
  ret.push_back("x");
  if (args->scale.defined()) {
    ret.push_back("scale");
  }
  ret.push_back("dy");
  return ret;
}

Attrs LayerNormDxSchema2Attrs(const LayerNormDxArgs* args) {
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = args->axis;
  attrs->epsilon = args->eps;
  if (args->scale.defined()) {
    attrs->set_scale_bias = true;
  } else {
    attrs->set_scale_bias = false;
  }
  return Attrs(attrs);
}

HashKey LayerNormDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const LayerNormDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->eps;
  return key;
}

RAF_TVM(layer_norm_dx, LayerNormDx, LayerNormDxArgs, LayerNormDxSchema2Args,
        LayerNormDxSchemaArgNames, LayerNormDxSchema2Attrs, LayerNormDxHasher, kOpaque);

std::vector<Value> BatchNormSchema2Args(const BatchNormArgs* args) {
  return {args->x, args->running_mean, args->running_var, args->w, args->b};
}

std::vector<std::string> BatchNormSchemaArgNames(const op::CallValues& call) {
  return {"x", "running_mean", "running_var", "w", "b"};
}

Attrs BatchNormSchema2Attrs(const BatchNormArgs* args) {
  auto attrs = make_object<BatchNormAttrs>();
  attrs->momentum = args->momentum;
  attrs->eps = args->eps;
  return Attrs(attrs);
}

HashKey BatchNormHasher(const std::vector<Type>& param_types, const Type& ret_type,
                        const BatchNormArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  key << args->momentum;
  key << args->eps;
  return key;
}

RAF_TVM(batch_norm_train, BatchNormTrain, BatchNormArgs, BatchNormSchema2Args,
        BatchNormSchemaArgNames, BatchNormSchema2Attrs, BatchNormHasher, kOpaque);
RAF_TVM(batch_norm_infer, BatchNormInfer, BatchNormArgs, BatchNormSchema2Args,
        BatchNormSchemaArgNames, BatchNormSchema2Attrs, BatchNormHasher, kOpaque);

std::vector<Value> BatchNormTrainDxwbSchema2Args(const BatchNormTrainDxwbArgs* args) {
  return {args->dy, args->x, args->w, args->b};
}

std::vector<std::string> BatchNormTrainDxwbSchemaArgNames(const op::CallValues& call) {
  return {"dy", "x", "w", "b"};
}

Attrs BatchNormTrainDxwbSchema2Attrs(const BatchNormTrainDxwbArgs* args) {
  auto attrs = make_object<BatchNormAttrs>();
  attrs->momentum = 0;  // momentum is not used in the gradient
  attrs->eps = args->eps;
  return Attrs(attrs);
}

HashKey BatchNormTrainDxwbHasher(const std::vector<Type>& param_types, const Type& ret_type,
                                 const BatchNormTrainDxwbArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  key << args->eps;
  return key;
}

RAF_TVM(batch_norm_train_dxwb, BatchNormTrainDxwb, BatchNormTrainDxwbArgs,
        BatchNormTrainDxwbSchema2Args, BatchNormTrainDxwbSchemaArgNames,
        BatchNormTrainDxwbSchema2Attrs, BatchNormTrainDxwbHasher, kOpaque);

std::vector<Value> ThresholdSchema2Args(const ThresholdArgs* args) {
  return {args->x};
}

std::vector<std::string> ThresholdSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ThresholdSchema2Attrs(const ThresholdArgs* args) {
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<ThresholdAttrs>();
  attrs->threshold = args->threshold;
  attrs->value = args->value;
  return Attrs(attrs);
}

HashKey ThresholdHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const ThresholdArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->threshold;
  key << args->value;
  return key;
}

RAF_TVM(threshold, Threshold, ThresholdArgs, ThresholdSchema2Args, ThresholdSchemaArgNames,
        ThresholdSchema2Attrs, ThresholdHasher, kElemWise);

std::vector<Value> ThresholdDxSchema2Args(const ThresholdDxArgs* args) {
  std::vector<Value> ret;
  ret.push_back(args->x);
  ret.push_back(args->dy);
  return ret;
}

std::vector<std::string> ThresholdDxSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<ThresholdDxArgs>();
  std::vector<std::string> ret;
  ret.push_back("x");
  ret.push_back("dy");
  return ret;
}

Attrs ThresholdDxSchema2Attrs(const ThresholdDxArgs* args) {
  auto attrs = make_object<ThresholdDxAttrs>();
  attrs->threshold = args->threshold;
  return Attrs(attrs);
}

HashKey ThresholdDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const ThresholdDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->threshold;
  return key;
}

RAF_TVM(threshold_dx, ThresholdDx, ThresholdDxArgs, ThresholdDxSchema2Args,
        ThresholdDxSchemaArgNames, ThresholdDxSchema2Attrs, ThresholdDxHasher, kElemWise);

std::vector<Value> PadSchema2Args(const PadArgs* args) {
  return {args->x};
}

std::vector<std::string> PadSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs PadSchema2Attrs(const PadArgs* args) {
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<PadAttrs>();
  Array<Array<Integer>> pad_width;
  for (int i = 0; i < args->pad_width.size(); i += 2) {
    Array<Integer> width{Integer(args->pad_width[i]), Integer(args->pad_width[i + 1])};
    pad_width.push_back(width);
  }
  attrs->pad_width = pad_width;
  attrs->pad_value = args->pad_value;
  attrs->pad_mode = args->pad_mode;
  return Attrs(attrs);
}

HashKey PadHasher(const std::vector<Type>& param_types, const Type& y_type, const PadArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->pad_width;
  key << args->pad_value;
  key << args->pad_mode;
  return key;
}

RAF_TVM(pad, Pad, PadArgs, PadSchema2Args, PadSchemaArgNames, PadSchema2Attrs, PadHasher,
        kInjective);

Array<tvm::te::Tensor> PadCompute(const Attrs& attrs, const Array<tvm::te::Tensor>& inputs,
                                  const Type& out_type) {
  const auto* param = attrs.as<PadAttrs>();
  ICHECK(param != nullptr);
  auto pad_width = param->pad_width;
  ICHECK(pad_width.size() == inputs[0].ndim() && pad_width[0].size() == 2) << "Illegal pad_width";
  Array<IndexExpr> pad_before;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_before.push_back(pad_width[i][0]);
  }
  Array<IndexExpr> pad_after;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_after.push_back(pad_width[i][1]);
  }
  return Array<tvm::te::Tensor>{tvm::topi::pad(
      inputs[0], pad_before, pad_after, tvm::tir::make_const(inputs[0]->dtype, param->pad_value),
      "T_pad", tvm::topi::kElementWise, param->pad_mode)};
}

RAF_REGISTER_OP("raf.op.tvm.pad").set_attr<tvm::relay::FTVMCompute>("FTVMCompute", PadCompute);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
