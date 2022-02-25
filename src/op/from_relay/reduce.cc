/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/reduce.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "./from_relay_utils.h"
#include "tvm/relay/attrs/reduce.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY("sum", "raf.op.sum",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    if (relay_attrs->axis.defined()) {
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    } else {
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(Array<Integer>())));
                    }
                    raf_args.push_back(
                        MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->keepdims)));
                    raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("mean", "raf.op.mean",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->keepdims)));
                    raf_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return raf_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace raf
