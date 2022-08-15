/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/nn.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "tvm/relay/attrs/nn.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_GENERIC_ATTR_OP_FROM_RELAY("nn.batch_matmul", "raf.op.batch_matmul_nt");
RAF_GENERIC_ATTR_OP_FROM_RELAY("nn.dense", "raf.op.dense");

RAF_OP_FROM_RELAY("nn.conv2d", "raf.op.conv2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<Conv2DAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));

                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but RAF currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->groups)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->kernel_layout)));
                    if (relay_attrs->out_layout != "") {
                      raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->out_layout)));
                    } else {
                      raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    }
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("nn.conv2d_transpose", "raf.op.conv2d_transpose",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<Conv2DTransposeAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));

                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but RAF currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->output_padding)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->groups)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->kernel_layout)));
                    if (relay_attrs->out_layout != "") {
                      raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->out_layout)));
                    } else {
                      raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    }
                    return raf_args;
                  });

#define RAF_SOFTMAX_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME)                                      \
  RAF_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME,                                                    \
                    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
                      Array<Expr> raf_args = args;                                                 \
                      const auto* relay_attrs = attrs.as<SoftmaxAttrs>();                          \
                      raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));      \
                      return raf_args;                                                             \
                    })

RAF_SOFTMAX_OP_FROM_RELAY("nn.softmax", "raf.op.softmax");
RAF_SOFTMAX_OP_FROM_RELAY("nn.log_softmax", "raf.op.log_softmax");

RAF_OP_FROM_RELAY("nn.bias_add", "raf.op.bias_add",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<BiasAddAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("nn.max_pool2d", "raf.op.max_pool2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<MaxPool2DAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pool_size)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));
                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but RAF currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->ceil_mode)));
                    raf_args.push_back(MakeConstant(BoolValue::make(true)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("nn.avg_pool2d", "raf.op.avg_pool2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<AvgPool2DAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pool_size)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));
                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but RAF currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->ceil_mode)));
                    raf_args.push_back(MakeConstant(BoolValue::make(true)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    return raf_args;
                  });

Array<Expr> AdaptivePoolFromRelay(const Attrs& attrs, const Array<Expr>& args,
                                  const VarValueMap& val_map) {
  Array<Expr> raf_args = args;
  const auto* relay_attrs = attrs.as<AdaptivePool2DAttrs>();
  raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->output_size)));
  raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
  return raf_args;
}

RAF_OP_FROM_RELAY("nn.adaptive_max_pool2d", "raf.op.adaptive_max_pool2d", AdaptivePoolFromRelay);
RAF_OP_FROM_RELAY("nn.adaptive_avg_pool2d", "raf.op.adaptive_avg_pool2d", AdaptivePoolFromRelay);

RAF_OP_FROM_RELAY("nn.layer_norm", "raf.op.layer_norm",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<LayerNormAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->epsilon)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("nn.batch_norm", "raf.op.batch_norm_train",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    const auto* relay_attrs = attrs.as<BatchNormAttrs>();
                    raf_args.push_back(args[0]);                               // x
                    raf_args.push_back(args[3]);                               // running_mean
                    raf_args.push_back(args[4]);                               // running_var
                    raf_args.push_back(args[1]);                               // w
                    raf_args.push_back(args[2]);                               // b
                    raf_args.push_back(MakeConstant(ScalarValue::make(0.1)));  // momentum
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->epsilon)));
                    return raf_args;
                  });

Array<Array<Expr>> BatchNormMutationFromRelay(const Var& var, const Call& call) {
  Array<Array<Expr>> res = {
      {TryGetMayShare(call->args[1]), TupleGetItem(var, 1)},  // running_mean
      {TryGetMayShare(call->args[2]), TupleGetItem(var, 2)}   // running_var
  };
  return res;
}

RAF_OP_MUTATION_FROM_RELAY("nn.batch_norm", BatchNormMutationFromRelay);

RAF_OP_FROM_RELAY("nn.pad", "raf.op.pad",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    raf_args.push_back(args[0]);
                    const auto* relay_attrs = attrs.as<PadAttrs>();
                    Array<Integer> flat_pad_width;
                    for (int i = 0; i < relay_attrs->pad_width.size(); ++i) {
                      for (int j = 0; j < relay_attrs->pad_width[i].size(); ++j) {
                        flat_pad_width.push_back(relay_attrs->pad_width[i][j]);
                      }
                    }
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(flat_pad_width)));
                    const auto* konst = GetKonstFromValueMap(args[1], val_map);
                    CHECK(konst) << "'pad_value' must be a const tensor.";
                    raf_args.push_back(MakeConstant(Constant2ScalarValue<double>(konst)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->pad_mode)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("nn.dropout", "raf.op._contrib_dropout",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<DropoutAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->rate)));
                    // We make up a NULL in_states argument which is required by raf but not pytorch
                    raf_args.push_back(MakeNull());
                    return raf_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace raf
