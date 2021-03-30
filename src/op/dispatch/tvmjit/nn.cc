/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/nn.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/nn.h>
#include <array>
#include "mnm/op_utils.h"
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/ufunc.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace tvm_attrs;
using namespace schema;

Attrs BinarySchema2DenseAttrs(const BinaryArgs* args) {
  auto attrs = make_object<tvm::relay::DenseAttrs>();
  return Attrs(attrs);
}

MNM_TVMJIT(BatchMatmul, "mnm.op.batch_matmul", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Matmul, "mnm.op.matmul", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           BinarySchema2DenseAttrs, GenericHasher);
MNM_TVMJIT(MatmulTN, "mnm.op.matmul_tn", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           BinarySchema2DenseAttrs, GenericHasher);
MNM_TVMJIT(MatmulNT, "mnm.op.matmul_nt", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           BinarySchema2DenseAttrs, GenericHasher);
MNM_TVMJIT(MatmulTT, "mnm.op.matmul_tt", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           BinarySchema2DenseAttrs, GenericHasher);
MNM_TVMJIT(Dense, "mnm.op.dense", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           BinarySchema2DenseAttrs, GenericHasher);

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
  attrs->kernel_size = NullValue<Array<tvm::relay::IndexExpr> >();
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

MNM_TVMJIT(Conv2d, "mnm.op.conv2d", ConvArgs, ConvSchema2Args, ConvSchemaArgNames, ConvSchema2Attrs,
           Conv2dHasher);

using tvm::relay::IndexExpr;
/*! \brief Attributes used in layer_norm operator */

struct Conv2dDxwAttrs : public tvm::AttrsNode<Conv2dDxwAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;
  bool use_output;
  TVM_DECLARE_ATTRS(Conv2dDxwAttrs, "relay.attrs.Conv2dDxwAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr> >());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(use_output).set_default(true).describe("whether use output to compute gradient");
  }
};

TVM_REGISTER_NODE_TYPE(Conv2dDxwAttrs);

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
  CHECK(args->shape.defined());
  auto shape = args->shape.value();
  for (int i = 0; i < shape.size(); ++i) {
    attrs->kernel_size.push_back(IntImm(tvm::runtime::DataType::Int(32), shape[i]->value));
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

MNM_TVMJIT_PLEVEL(Conv2dDx, "mnm.op.conv2d_dx", ConvDxwArgs, ConvDxwSchema2Args,
                  ConvDxwSchemaArgNames, ConvDxwSchema2Attrs, Conv2dDxwHasher, 9);
MNM_TVMJIT_PLEVEL(Conv2dDw, "mnm.op.conv2d_dw", ConvDxwArgs, ConvDxwSchema2Args,
                  ConvDxwSchemaArgNames, ConvDxwSchema2Attrs, Conv2dDxwHasher, 9);

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

MNM_TVMJIT(Softmax, "mnm.op.softmax", SoftmaxArgs, SoftmaxSchema2Args, SoftmaxSchemaArgNames,
           SoftmaxSchema2Attrs, SoftmaxHasher);
MNM_TVMJIT(SoftmaxDx, "mnm.op.softmax_dx", SoftmaxDxArgs, SoftmaxDxSchema2Args,
           SoftmaxDxSchemaArgNames, SoftmaxDxSchema2Attrs, SoftmaxDxHasher);
MNM_TVMJIT(LogSoftmax, "mnm.op.log_softmax", SoftmaxArgs, SoftmaxSchema2Args, SoftmaxSchemaArgNames,
           SoftmaxSchema2Attrs, SoftmaxHasher);
MNM_TVMJIT(LogSoftmaxDx, "mnm.op.log_softmax_dx", SoftmaxDxArgs, SoftmaxDxSchema2Args,
           SoftmaxDxSchemaArgNames, SoftmaxDxSchema2Attrs, SoftmaxDxHasher);

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

MNM_TVMJIT(BiasAdd, "mnm.op.bias_add", BiasAddArgs, BiasAddSchema2Args, BiasAddSchemaArgNames,
           BiasAddSchema2Attrs, BiasAddHasher);

std::vector<Value> ContribDropoutSchema2Args(const DropoutArgs* args) {
  return {args->x};
}

std::vector<std::string> ContribDropoutSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ContribDropoutSchema2Attrs(const DropoutArgs* args) {
  auto attrs = make_object<tvm::relay::DropoutAttrs>();
  attrs->rate = args->p;
  return Attrs(attrs);
}

HashKey ContribDropoutHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const DropoutArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->p;
  return key;
}

MNM_TVMJIT(Dropout, "mnm.op._contrib_dropout", DropoutArgs, ContribDropoutSchema2Args,
           ContribDropoutSchemaArgNames, ContribDropoutSchema2Attrs, ContribDropoutHasher);

template <typename T>
std::vector<Value> PoolSchema2Args(const T* args) {
  return {args->x};
}

std::vector<std::string> PoolSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs MaxPoolSchema2Attrs(const PoolArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
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
  attrs->layout = args->layout;
  CHECK_EQ(args->include_pad, true);
  return Attrs(attrs);
}

Attrs AvgPoolSchema2Attrs(const PoolArgs* args) {
  std::vector<int64_t> stride = Pad<2>(args->stride);
  std::vector<int64_t> padding = args->padding.size() > 1 ? args->padding : Pad<2>(args->padding);
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
  attrs->layout = args->layout;
  return Attrs(attrs);
}

HashKey PoolHasher(const std::vector<Type>& param_types, const Type& y_type, const PoolArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->stride;
  key << args->padding;
  key << args->kernel;
  key << args->ceil_mode;
  key << args->include_pad;
  return key;
}

MNM_TVMJIT(MaxPool2D, "mnm.op.max_pool2d", PoolArgs, PoolSchema2Args<PoolArgs>, PoolSchemaArgNames,
           MaxPoolSchema2Attrs, PoolHasher);
MNM_TVMJIT(AvgPool2D, "mnm.op.avg_pool2d", PoolArgs, PoolSchema2Args<PoolArgs>, PoolSchemaArgNames,
           AvgPoolSchema2Attrs, PoolHasher);

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

MNM_TVMJIT(AdaptiveMaxPool2D, "mnm.op.adaptive_max_pool2d", AdaptivePoolArgs,
           PoolSchema2Args<AdaptivePoolArgs>, PoolSchemaArgNames, AdaptivePoolSchema2Attrs,
           AdaptivePoolHasher);
MNM_TVMJIT(AdaptiveAvgPool2D, "mnm.op.adaptive_avg_pool2d", AdaptivePoolArgs,
           PoolSchema2Args<AdaptivePoolArgs>, PoolSchemaArgNames, AdaptivePoolSchema2Attrs,
           AdaptivePoolHasher);

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

MNM_TVMJIT(AvgPool2DDx, "mnm.op.avg_pool2d_dx", PoolDxArgs, PoolDxSchema2Args<PoolDxArgs>,
           PoolDxSchemaArgNames, AvgPoolDxSchema2Attrs, PoolDxHasher);
MNM_TVMJIT(MaxPool2DDx, "mnm.op.max_pool2d_dx", PoolDxArgs, PoolDxSchema2Args<PoolDxArgs>,
           PoolDxSchemaArgNames, MaxPoolDxSchema2Attrs, PoolDxHasher);

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

MNM_TVMJIT(AdaptiveAvgPool2DDx, "mnm.op.adaptive_avg_pool2d_dx", AdaptivePoolDxArgs,
           PoolDxSchema2Args<AdaptivePoolDxArgs>, PoolDxSchemaArgNames,
           AdaptiveAvgPoolDxSchema2Attrs, AdaptivePoolDxHasher);
MNM_TVMJIT(AdaptiveMaxPool2DDx, "mnm.op.adaptive_max_pool2d_dx", AdaptivePoolDxArgs,
           PoolDxSchema2Args<AdaptivePoolDxArgs>, PoolDxSchemaArgNames,
           AdaptiveMaxPoolDxSchema2Attrs, AdaptivePoolDxHasher);

struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  int axis;
  double epsilon;
  bool set_scale_bias;
  TVM_DECLARE_ATTRS(LayerNormAttrs, "relay.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Specify which shape axis denotes the channel.");
    TVM_ATTR_FIELD(epsilon).set_default(1e-5).describe(
        "Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(set_scale_bias)
        .set_default(true)
        .describe("whether set scale and bias for layer norm");
  }
};  // struct LayerNormAttrs

TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

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

MNM_TVMJIT(LayerNorm, "mnm.op.layer_norm", LayerNormArgs, LayerNormSchema2Args,
           LayerNormSchemaArgNames, LayerNormSchema2Attrs, LayerNormHasher);

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

MNM_TVMJIT(LayerNormDx, "mnm.op.layer_norm_dx", LayerNormDxArgs, LayerNormDxSchema2Args,
           LayerNormDxSchemaArgNames, LayerNormDxSchema2Attrs, LayerNormDxHasher);

struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  double momentum;
  double eps;
  TVM_DECLARE_ATTRS(BatchNormAttrs, "attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(momentum);
    TVM_ATTR_FIELD(eps);
  }
};
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

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

MNM_TVMJIT(BatchNormTrain, "mnm.op.batch_norm_train", BatchNormArgs, BatchNormSchema2Args,
           BatchNormSchemaArgNames, BatchNormSchema2Attrs, BatchNormHasher);
MNM_TVMJIT(BatchNormInfer, "mnm.op.batch_norm_infer", BatchNormArgs, BatchNormSchema2Args,
           BatchNormSchemaArgNames, BatchNormSchema2Attrs, BatchNormHasher);

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

MNM_TVMJIT(BatchNormTrainDxwb, "mnm.op.batch_norm_train_dxwb", BatchNormTrainDxwbArgs,
           BatchNormTrainDxwbSchema2Args, BatchNormTrainDxwbSchemaArgNames,
           BatchNormTrainDxwbSchema2Attrs, BatchNormTrainDxwbHasher);

std::vector<Value> PadSchema2Args(const PadArgs* args) {
  return {args->x};
}

std::vector<std::string> PadSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs PadSchema2Attrs(const PadArgs* args) {
  // attrs will be later passed to compute & schedule functions
  auto attrs = make_object<tvm::relay::PadAttrs>();
  Array<Array<Integer> > pad_width;
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

MNM_TVMJIT(Pad, "mnm.op.pad", PadArgs, PadSchema2Args, PadSchemaArgNames, PadSchema2Attrs,
           PadHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
