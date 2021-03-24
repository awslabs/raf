/*!
 *  * Copyright (c) 2021 by Contributors
 *   * \file ./src/op/from_relay/reduce.cc
 *    * \brief Operators bridged from Relay.
 *     */
#include "./from_relay_utils.h"
#include "tvm/relay/attrs/reduce.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("sum", "mnm.op.sum", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<ReduceAttrs>();
  mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->keepdims)));
  CHECK(relay_attrs->exclude == false) << "Do not support exclude for sum";
  return mnm_args;
});

}  // namespace from_relay
}  // namespace op
}  // namespace mnm