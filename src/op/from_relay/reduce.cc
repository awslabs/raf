/*!
 *  * Copyright (c) 2021 by Contributors
 *   * \file ./src/op/from_relay/reduce.cc
 *    * \brief Operators bridged from Relay.
 *     */
#include "mnm/op_utils.h"
#include "./from_relay_utils.h"
#include "tvm/relay/attrs/reduce.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("sum", "mnm.op.sum",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    if (relay_attrs->axis.defined()) {
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    } else {
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(Array<Integer>())));
                    }
                    mnm_args.push_back(
                        MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->keepdims)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("mean", "mnm.op.mean",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->keepdims)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return mnm_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
