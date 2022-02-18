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
 * \file ./src/op/from_relay/reduce.cc
 * \brief Operators bridged from Relay.
 */
#include "mnm/op_utils.h"
#include "./from_relay_utils.h"
#include "tvm/relay/attrs/reduce.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("sum", "mnm.op.sum",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    if (relay_attrs->axis.defined()) {
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    } else {
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(Array<Integer>())));
                    }
                    mnm_args.push_back(
                        MakeConstant(IntValue::make(DataType::Int(64), relay_attrs->keepdims)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("mean", "mnm.op.mean",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ReduceAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->keepdims)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->exclude)));
                    return mnm_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
