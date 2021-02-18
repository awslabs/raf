/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/unary.cc
 * \brief Operators bridged from Relay.
 */
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_GENERIC_ATTR_OP_FROM_RELAY("copy", "mnm.op.copy");
MNM_GENERIC_ATTR_OP_FROM_RELAY("abs", "mnm.op.abs");
MNM_GENERIC_ATTR_OP_FROM_RELAY("ceil", "mnm.op.ceil");
MNM_GENERIC_ATTR_OP_FROM_RELAY("floor", "mnm.op.floor");
MNM_GENERIC_ATTR_OP_FROM_RELAY("log", "mnm.op.log");
MNM_GENERIC_ATTR_OP_FROM_RELAY("exp", "mnm.op.exp");
MNM_GENERIC_ATTR_OP_FROM_RELAY("cos", "mnm.op.cos");
MNM_GENERIC_ATTR_OP_FROM_RELAY("sin", "mnm.op.sin");
MNM_GENERIC_ATTR_OP_FROM_RELAY("sign", "mnm.op.sign");
MNM_GENERIC_ATTR_OP_FROM_RELAY("round", "mnm.op.round");
MNM_GENERIC_ATTR_OP_FROM_RELAY("nn.relu", "mnm.op.relu");
MNM_GENERIC_ATTR_OP_FROM_RELAY("erf", "mnm.op.erf");
MNM_GENERIC_ATTR_OP_FROM_RELAY("sqrt", "mnm.op.sqrt");
MNM_GENERIC_ATTR_OP_FROM_RELAY("rsqrt", "mnm.op.rsqrt");
MNM_GENERIC_ATTR_OP_FROM_RELAY("atan", "mnm.op.atan");
MNM_GENERIC_ATTR_OP_FROM_RELAY("negative", "mnm.op.negative");
MNM_GENERIC_ATTR_OP_FROM_RELAY("sigmoid", "mnm.op.sigmoid");
MNM_GENERIC_ATTR_OP_FROM_RELAY("tanh", "mnm.op.tanh");
MNM_GENERIC_ATTR_OP_FROM_RELAY("nn.batch_flatten", "mnm.op.batch_flatten");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
