/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/algorithm.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/algorithm.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY(
    "topk", "raf.op.topk",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
      Array<Expr> raf_args = args;
      const auto* relay_attrs = attrs.as<TopKAttrs>();
      raf_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->k.value())));
      raf_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->axis)));
      raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->ret_type)));
      raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->is_ascend)));
      raf_args.push_back(
          MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
      return raf_args;
    });

}  // namespace from_relay
}  // namespace op
}  // namespace raf