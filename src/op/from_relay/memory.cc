/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/memory.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/device_copy.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("device_copy", "mnm.op.device_copy",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<DeviceCopyAttrs>();
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->src_dev_type)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->dst_dev_type)));
                    return mnm_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
