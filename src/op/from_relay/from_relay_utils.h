/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/from_relay_utils.h
 * \brief Utility methods for Relay to Meta op conversion.
 */
#pragma once
#include "mnm/op.h"
#include "mnm/ir.h"

namespace mnm {
namespace op {
namespace from_relay {

using namespace mnm::ir;
using namespace mnm::value;
using namespace tvm;
using namespace ::tvm::relay;

#define MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME, RELAY_2_MNM_ARGS)                    \
  RELAY_REGISTER_OP(RELAY_OP_NAME)                                                         \
      .set_attr<op::FMNMFromRelay>("FMNMFromRelay",                                        \
                                   [](const Attrs& attrs, const Array<Expr>& args) {       \
                                     static const Op& op = Op::Get(MNM_OP_NAME);           \
                                     Array<Expr> mnm_args = RELAY_2_MNM_ARGS(attrs, args); \
                                     return Call(op, mnm_args);                            \
                                   })

#define MNM_BINARY_UFUNC_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME)                                 \
  MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME, [&](const Attrs& attrs, const Array<Expr>& args) { \
    Array<Expr> mnm_args = args;                                                                   \
    mnm_args.push_back(MakeConstant(NullValue<Value>()));                                          \
    mnm_args.push_back(MakeConstant(NullValue<Value>()));                                          \
    return mnm_args;                                                                               \
  })

#define MNM_GENERIC_ATTR_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME) \
  MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME,                    \
                    [&](const Attrs& attrs, const Array<Expr>& args) { return args; })

std::vector<int64_t> ArrayToInt(const Array<IndexExpr>& arr);
std::vector<int64_t> ArrayToInt(const Array<Integer>& arr);
std::vector<int64_t> ArrayToInt(const ArrayNode& arr);
TupleValue ArrayToIntTuple(const Array<IndexExpr>& arr);
TupleValue ArrayToIntTuple(const Array<Integer>& arr);
TupleValue ArrayToIntTuple(const std::vector<int64_t>& arr);
TupleValue ArrayToIntTuple(const ArrayNode& arr);

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
