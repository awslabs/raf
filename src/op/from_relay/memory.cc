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
                    auto src_device_str =
                        std::string(Device(relay_attrs->src_virtual_device->ToDevice()).c_str());
                    mnm_args.push_back(MakeConstant(StringValue::make(src_device_str)));
                    auto dst_device_str =
                        std::string(Device(relay_attrs->dst_virtual_device->ToDevice()).c_str());
                    mnm_args.push_back(MakeConstant(StringValue::make(dst_device_str)));
                    return mnm_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
