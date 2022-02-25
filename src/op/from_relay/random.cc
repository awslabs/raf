/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/random.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "tvm/relay/attrs/random.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY("random.threefry_generate", "raf.op.threefry_generate",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ThreefryGenerateAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->out_shape)));
                    return raf_args;
                  });

RAF_GENERIC_ATTR_OP_FROM_RELAY("random.threefry_split", "raf.op.threefry_split");

}  // namespace from_relay
}  // namespace op
}  // namespace raf
