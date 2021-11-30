/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/from_relay_utils.h
 * \brief Utility methods for Relay to Meta op conversion.
 */
#pragma once
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/tensor.h"

namespace mnm {
namespace op {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::tensor;
using namespace ::tvm::relay;

using VarValueMap = Map<Var, Expr>;

#define MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME, RELAY_2_MNM_ARGS)                 \
  RELAY_REGISTER_OP(RELAY_OP_NAME)                                                      \
      .set_attr<op::FMNMFromRelay>(                                                     \
          "FMNMFromRelay",                                                              \
          [](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
            static const Op& op = Op::Get(MNM_OP_NAME);                                 \
            Array<Expr> mnm_args = RELAY_2_MNM_ARGS(attrs, args, val_map);              \
            return Call(op, mnm_args);                                                  \
          })

#define MNM_OP_MUTATION_FROM_RELAY(RELAY_OP_NAME, MNM_OP_MUTATION) \
  RELAY_REGISTER_OP(RELAY_OP_NAME)                                 \
      .set_attr<op::FMNMMutationFromRelay>("FMNMMutationFromRelay", MNM_OP_MUTATION)

#define MNM_GENERIC_ATTR_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME)                                 \
  MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME,                                                    \
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
}  // namespace mnm
