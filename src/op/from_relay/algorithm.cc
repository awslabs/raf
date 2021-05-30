/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/from_relay/algorithm.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/algorithm.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY(
    "topk", "mnm.op.topk",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
      Array<Expr> mnm_args = args;
      const auto* relay_attrs = attrs.as<TopKAttrs>();
      mnm_args.push_back(args[0]);
      mnm_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->k.value())));
      mnm_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->axis)));
      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->ret_type)));
      mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->is_ascend)));
      mnm_args.push_back(
          MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
      return mnm_args;
    });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm