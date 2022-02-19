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
 * \file src/impl/dataflow_pattern.cc
 * \brief Dataflow pattern of Meta IR
 */

#include <tvm/node/repr_printer.h>

#include "mnm/registry.h"
#include "mnm/dataflow_pattern.h"

namespace mnm {
namespace ir {

using tvm::ReprPrinter;

// Pattern sugars
DFPattern IsRelayConstant() {
  return RelayConstantPattern(make_object<RelayConstantPatternNode>());
}

DFPattern IsConstant(ObjectRef value) {
  ObjectPtr<ConstantPatternNode> n = make_object<ConstantPatternNode>();
  n->value = std::move(value);
  return RelayConstantPattern(n);
}

MNM_REGISTER_GLOBAL("mnm.ir.dataflow_pattern.is_constant").set_body_typed(IsConstant);

}  // namespace ir
}  // namespace mnm
