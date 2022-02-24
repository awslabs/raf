/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/unary.cc
 * \brief Operators bridged from Relay.
 */
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_GENERIC_ATTR_OP_FROM_RELAY("copy", "raf.op.copy");
RAF_GENERIC_ATTR_OP_FROM_RELAY("abs", "raf.op.abs");
RAF_GENERIC_ATTR_OP_FROM_RELAY("ceil", "raf.op.ceil");
RAF_GENERIC_ATTR_OP_FROM_RELAY("floor", "raf.op.floor");
RAF_GENERIC_ATTR_OP_FROM_RELAY("log", "raf.op.log");
RAF_GENERIC_ATTR_OP_FROM_RELAY("exp", "raf.op.exp");
RAF_GENERIC_ATTR_OP_FROM_RELAY("cos", "raf.op.cos");
RAF_GENERIC_ATTR_OP_FROM_RELAY("sin", "raf.op.sin");
RAF_GENERIC_ATTR_OP_FROM_RELAY("sign", "raf.op.sign");
RAF_GENERIC_ATTR_OP_FROM_RELAY("round", "raf.op.round");
RAF_GENERIC_ATTR_OP_FROM_RELAY("nn.relu", "raf.op.relu");
RAF_GENERIC_ATTR_OP_FROM_RELAY("erf", "raf.op.erf");
RAF_GENERIC_ATTR_OP_FROM_RELAY("sqrt", "raf.op.sqrt");
RAF_GENERIC_ATTR_OP_FROM_RELAY("rsqrt", "raf.op.rsqrt");
RAF_GENERIC_ATTR_OP_FROM_RELAY("atan", "raf.op.atan");
RAF_GENERIC_ATTR_OP_FROM_RELAY("negative", "raf.op.negative");
RAF_GENERIC_ATTR_OP_FROM_RELAY("sigmoid", "raf.op.sigmoid");
RAF_GENERIC_ATTR_OP_FROM_RELAY("tanh", "raf.op.tanh");
RAF_GENERIC_ATTR_OP_FROM_RELAY("nn.batch_flatten", "raf.op.batch_flatten");

}  // namespace from_relay
}  // namespace op
}  // namespace raf
