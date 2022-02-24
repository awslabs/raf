/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/binary.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/ir.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

using namespace raf::ir;

#define RAF_BINARY_UFUNC_ATTR_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME)                            \
  RAF_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME,                                                    \
                    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
                      Array<Expr> raf_args = args;                                                 \
                      raf_args.push_back(MakeConstant(NullValue<Value>()));                        \
                      raf_args.push_back(MakeConstant(NullValue<Value>()));                        \
                      return raf_args;                                                             \
                    })

RAF_BINARY_UFUNC_ATTR_OP_FROM_RELAY("add", "raf.op.add");
RAF_BINARY_UFUNC_ATTR_OP_FROM_RELAY("subtract", "raf.op.subtract");
RAF_GENERIC_ATTR_OP_FROM_RELAY("divide", "raf.op.divide");
RAF_GENERIC_ATTR_OP_FROM_RELAY("floor_divide", "raf.op.floor_divide");
RAF_GENERIC_ATTR_OP_FROM_RELAY("multiply", "raf.op.multiply");
RAF_GENERIC_ATTR_OP_FROM_RELAY("power", "raf.op.power");
RAF_GENERIC_ATTR_OP_FROM_RELAY("mod", "raf.op.mod");
RAF_GENERIC_ATTR_OP_FROM_RELAY("less", "raf.op.less");
RAF_GENERIC_ATTR_OP_FROM_RELAY("greater", "raf.op.greater");
RAF_GENERIC_ATTR_OP_FROM_RELAY("less_equal", "raf.op.less_equal");
RAF_GENERIC_ATTR_OP_FROM_RELAY("greater_equal", "raf.op.greater_equal");
RAF_GENERIC_ATTR_OP_FROM_RELAY("equal", "raf.op.equal");
RAF_GENERIC_ATTR_OP_FROM_RELAY("not_equal", "raf.op.not_equal");
RAF_GENERIC_ATTR_OP_FROM_RELAY("maximum", "raf.op.maximum");
RAF_GENERIC_ATTR_OP_FROM_RELAY("minimum", "raf.op.minimum");
RAF_GENERIC_ATTR_OP_FROM_RELAY("logical_and", "raf.op.logical_and");

}  // namespace from_relay
}  // namespace op
}  // namespace raf
