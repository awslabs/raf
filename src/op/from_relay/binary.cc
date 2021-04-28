/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/binary.cc
 * \brief Operators bridged from Relay.
 */
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_GENERIC_ATTR_OP_FROM_RELAY("add", "mnm.op.add");
MNM_GENERIC_ATTR_OP_FROM_RELAY("subtract", "mnm.op.subtract");
MNM_GENERIC_ATTR_OP_FROM_RELAY("divide", "mnm.op.divide");
MNM_GENERIC_ATTR_OP_FROM_RELAY("multiply", "mnm.op.multiply");
MNM_GENERIC_ATTR_OP_FROM_RELAY("power", "mnm.op.power");
MNM_GENERIC_ATTR_OP_FROM_RELAY("mod", "mnm.op.mod");
MNM_GENERIC_ATTR_OP_FROM_RELAY("less", "mnm.op.less");
MNM_GENERIC_ATTR_OP_FROM_RELAY("greater", "mnm.op.greater");
MNM_GENERIC_ATTR_OP_FROM_RELAY("less_equal", "mnm.op.less_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("greater_equal", "mnm.op.greater_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("equal", "mnm.op.equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("not_equal", "mnm.op.not_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("maximum", "mnm.op.maximum");
MNM_GENERIC_ATTR_OP_FROM_RELAY("minimum", "mnm.op.minimum");
MNM_GENERIC_ATTR_OP_FROM_RELAY("logical_and", "mnm.op.logical_and");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
