/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file nn.h
 * \brief Extra TVM attributes for nn operators
 */
#pragma once
#include <tvm/ir/attrs.h>
#include <raf/ir.h>

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;

/*! \brief Attributes used in conv2d_dxw operator */
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
        .set_default(NullValue<Array<IndexExpr>>());
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

/*! \brief Attributes used in conv2d_transpose operator */
struct Conv2dTransposeDxwAttrs : public tvm::AttrsNode<Conv2dTransposeDxwAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;
  bool use_output;
  TVM_DECLARE_ATTRS(Conv2dTransposeDxwAttrs, "relay.attrs.Conv2dTransposeDxwAttrs") {
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
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "Zero-padding added to one side of the output."
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
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IOHW")
        .describe(
            "Dimension ordering of weight. Can be 'IOHW', 'IOHW16o16i', etc."
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

/*! \brief Attributes used in layer_norm operator */
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
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  double momentum;
  double eps;
  TVM_DECLARE_ATTRS(BatchNormAttrs, "attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(momentum);
    TVM_ATTR_FIELD(eps);
  }
};

/*! \brief Attributes used in pad operator */
struct PadAttrs : public tvm::AttrsNode<PadAttrs> {
  double pad_value;
  Array<Array<Integer>> pad_width;
  std::string pad_mode;

  TVM_DECLARE_ATTRS(PadAttrs, "raf.attrs.PadAttrs") {
    TVM_ATTR_FIELD(pad_value).set_default(0.0).describe(
        "The value used for padding when mode is 'constant'.");
    TVM_ATTR_FIELD(pad_width).describe(
        "Number of values padded to the edges of each axis, "
        "in the format of ((before_1, after_1), ..., (before_N, after_N))");
    TVM_ATTR_FIELD(pad_mode)
        .set_default("constant")
        .describe(
            "Padding type to use. \"constant\" pads with constant_value, "
            "\"edge\" pads using the edge values of the input array, "
            "\"reflect\" pads by reflecting values with respect to the edges.");
  }
};

/*! \brief Attributes used in threshold operator */
struct ThresholdAttrs : public tvm::AttrsNode<ThresholdAttrs> {
  double threshold;
  double value;
  TVM_DECLARE_ATTRS(ThresholdAttrs, "raf.attrs.ThresholdAttrs") {
    TVM_ATTR_FIELD(threshold).set_default(0.0).describe("The value to threshold at");
    TVM_ATTR_FIELD(value).set_default(0.0).describe("The value to replace with");
  }
};

/*! \brief Attributes used in threshold_dx operator */
struct ThresholdDxAttrs : public tvm::AttrsNode<ThresholdDxAttrs> {
  double threshold;

  TVM_DECLARE_ATTRS(ThresholdDxAttrs, "relay.attrs.ThresholdDxAttrs") {
    TVM_ATTR_FIELD(threshold).set_default(0.0).describe("The value to threshold at");
  }
};

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
