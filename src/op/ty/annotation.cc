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
 * \file src/op/ty/annotation.cc
 * \brief Typing of annotation operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/annotation.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace schema;

Type CompilerInfer(const CallValues& value) {
  const auto* args = value->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.compiler_begin", "Compiler", CompilerInfer);
MNM_OP_TYPE("mnm.op.compiler_end", "Compiler", CompilerInfer);

}  // namespace op
}  // namespace mnm
