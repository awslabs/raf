/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/from_relay_utils.cc
 * \brief Utility methods for Relay to Meta op conversion.
 */
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

using namespace mnm::value;

const ConstantNode* GetKonstFromValueMap(const Expr& expr, const VarValueMap& val_map) {
  ICHECK(expr->IsInstance<VarNode>()) << "Assume ANF!";
  auto var = Downcast<Var>(expr);
  ICHECK_EQ(val_map.count(var), 1) << "Cannot find the value of constant var " << var->name_hint()
                                   << " in value map. Maybe the IR is not in ANF?";
  const auto* konst = val_map[var].as<ConstantNode>();
  return konst;
}

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
