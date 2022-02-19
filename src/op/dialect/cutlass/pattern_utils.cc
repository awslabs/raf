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
 * \file src/op/dialect/cutlass/pattern_utils.cc
 * \brief dataflow pattern helpers
 */
#include "./pattern_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::ir;

DFPattern IsOps(std::vector<std::string> ops) {
  CHECK_GE(ops.size(), 1U);
  auto op = IsOp(ops[0]);
  for (const auto& name : ops) {
    op = op || IsOp(name);
  }
  return op;
}

EpilogueKindExt GetEpilogueKind(const Op& op) {
  if (!op.defined()) {
    return EpilogueKindExt::kLinearCombination;
  }
  const static std::unordered_map<Op, EpilogueKindExt, ObjectPtrHash, ObjectPtrEqual> epilogue_map =
      {{Op::Get("mnm.op.cutlass.relu"), EpilogueKindExt::kLinearCombinationRelu},
       {Op::Get("mnm.op.cutlass.gelu"), EpilogueKindExt::kLinearCombinationGelu}};
  auto it = epilogue_map.find(op);
  if (it == epilogue_map.end()) {
    LOG(FATAL) << "Unknown epilogue op: " << op->name;
  }
  return it->second;
}

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
