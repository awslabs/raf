/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/from_relay_utils.h
 * \brief Utility methods for Relay to RAF op conversion.
 */
#pragma once
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/tensor.h"

namespace raf {
namespace op {
namespace from_relay {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::tensor;
using namespace ::tvm::relay;

using VarValueMap = Map<Var, Expr>;

#define RAF_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME, RELAY_2_RAF_ARGS)                 \
  RELAY_REGISTER_OP(RELAY_OP_NAME)                                                      \
      .set_attr<op::FRAFFromRelay>(                                                     \
          "FRAFFromRelay",                                                              \
          [](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
            static const Op& op = Op::Get(RAF_OP_NAME);                                 \
            Array<Expr> raf_args = RELAY_2_RAF_ARGS(attrs, args, val_map);              \
            return Call(op, raf_args);                                                  \
          })

#define RAF_OP_MUTATION_FROM_RELAY(RELAY_OP_NAME, RAF_OP_MUTATION) \
  RELAY_REGISTER_OP(RELAY_OP_NAME)                                 \
      .set_attr<op::FRAFMutationFromRelay>("FRAFMutationFromRelay", RAF_OP_MUTATION)

#define RAF_GENERIC_ATTR_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME)                                 \
  RAF_OP_FROM_RELAY(RELAY_OP_NAME, RAF_OP_NAME,                                                    \
                    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
                      return args;                                                                 \
                    })

template <typename T>
ScalarValue Constant2ScalarValue(const ConstantNode* op) {
  T data = GetScalarValueData<T>(Downcast<TensorValue>(op->value));
  return ScalarValue::make(data);
}

const ConstantNode* GetKonstFromValueMap(const Expr& expr, const VarValueMap& val_map);

}  // namespace from_relay
}  // namespace op
}  // namespace raf
