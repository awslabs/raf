/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
