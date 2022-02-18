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
 * \file ./src/op/from_relay/random.cc
 * \brief Operators bridged from Relay.
 */
#include "mnm/op_utils.h"
#include "tvm/relay/attrs/random.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("random.threefry_generate", "mnm.op.threefry_generate",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ThreefryGenerateAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->out_shape)));
                    return mnm_args;
                  });

MNM_GENERIC_ATTR_OP_FROM_RELAY("random.threefry_split", "mnm.op.threefry_split");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
