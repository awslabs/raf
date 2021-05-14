/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/from_relay/random.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/random.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("random.threefry_generate", "mnm.op.threefry_generate",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ThreefryGenerateAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->out_shape)));
                    return mnm_args;
                  });

MNM_GENERIC_ATTR_OP_FROM_RELAY("random.threefry_split", "mnm.op.threefry_split");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
