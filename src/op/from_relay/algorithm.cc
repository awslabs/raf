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
 * \file ./src/op/from_relay/algorithm.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/algorithm.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY(
    "topk", "mnm.op.topk",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
      Array<Expr> mnm_args = args;
      const auto* relay_attrs = attrs.as<TopKAttrs>();
      mnm_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->k.value())));
      mnm_args.push_back(MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->axis)));
      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->ret_type)));
      mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->is_ascend)));
      mnm_args.push_back(
          MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
      return mnm_args;
    });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm